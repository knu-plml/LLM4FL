import codecs
import os
import numpy as np
import sys
import json
import subprocess

CODEGEN_FINETUNE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = CODEGEN_FINETUNE_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_codegen_finetune_input(buggy_file, rem_start, rem_end, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.finetuning.FineTuningData', 'inference',
        buggy_file, str(rem_start), str(rem_end), tmp_file
    ])

def codegen_finetune_input(output_file, humaneval_dir):
    loc_fp = codecs.open(CODEGEN_FINETUNE_DIR + '../../humaneval/humaneval_loc.txt', 'r', 'utf-8')
    codegen_input = {'config': 'finetune', 'data': {}}
    for line in loc_fp.readlines():
        filename, rem_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = CODEGEN_FINETUNE_DIR + '../../humaneval/tmp.json'
        get_codegen_finetune_input(humaneval_dir + 'src/main/java/humaneval/buggy/' + filename + '.java', start, end, tmp_file)
        
        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        elems = ['buggy function before', 'buggy line', 'buggy function after']
        inputs, outputs = '', ''
        number = 1

        for elem in elems:
            for line in result[elem].split('\n')[:-1]:
                inputs += str(number) + '\t' + line + '\n'
                if elem == 'buggy line':
                    outputs += str(number) + '\t' + line + '\n'
                number += 1

        codegen_input['data'][filename] = {
            'loc': rem_loc,
            'input': inputs,
            'gold_output': outputs
        }
        command(['rm', '-rf', tmp_file])
    json.dump(codegen_input, open(CODEGEN_FINETUNE_DIR + output_file, 'w'), indent=2)


def codegen_finetune_output(input_file, output_file, iterN, model_dir, model_name, num_output=10):
    tokenizer = AutoTokenizer.from_pretrained(CODEGEN_FINETUNE_DIR + '/'.join(model_dir.split('/')[:-2]) + '/' + model_name[:-9])
    model = CodeGenForCausalLM.from_pretrained(CODEGEN_FINETUNE_DIR + model_dir + model_name + '/' + iterN).to(device_ids[0])
    
    codegen_output = json.load(open(CODEGEN_FINETUNE_DIR + input_file, 'r'))
    codegen_output['model'] = model_name
    for i, filename in enumerate(codegen_output['data']):
        text = codegen_output['data'][filename]['input']

        print(i + 1, 'generating', filename)

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device_ids[0])
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        if input_ids.size(1) >= 768:
            print('too long:', input_ids.size(1))
            continue

        try:
            generated_ids = model.generate(
                input_ids, max_new_tokens=64, num_beams=10, num_return_sequences=num_output, early_stopping=True, 
                pad_token_id=eos_id, eos_token_id=eos_id
            )
        except Exception as e:
            print(e)
            continue

        output = []
        for generated_id in generated_ids:
            output.append(tokenizer.decode(generated_id, skip_special_tokens=False))
        codegen_output['data'][filename]['output'] = output
    json.dump(codegen_output, open(CODEGEN_FINETUNE_DIR + output_file, 'w'), indent=2)

if __name__ == '__main__':
    model_dir = '../../models/noPretrained-model/'
    model_name = 'codegen-350M-finetune'
    humaneval_dir = CODEGEN_FINETUNE_DIR + '../../humaneval-java/'
    
    input_dir = './humaneval_result/'
    input_file = input_dir + 'codegen_input.json'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(input_file):
        print("==========Preparing input of benchmark to finetuned codegen model==========")
        codegen_finetune_input(input_file, humaneval_dir=humaneval_dir)
        print("==========Input written to " + input_file)
    
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    device_ids = [0]

    import torch
    from transformers import AutoTokenizer, CodeGenForCausalLM

    for i in range(5):
        output_file = 'humaneval_result/codegen_output' + str(i) + '.json'
        print("==========Generating output of benchmark by " + model_name + "==========")
        codegen_finetune_output(input_file, output_file, str(i), model_dir, model_name, num_output=10)
        print("==========Output written to " + output_file)
