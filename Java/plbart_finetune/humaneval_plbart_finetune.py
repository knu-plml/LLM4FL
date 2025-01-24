import re
import os
import sys
import json
import codecs
import numpy as np
import subprocess

PLBART_FINETUNE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = PLBART_FINETUNE_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_plbart_finetune_input(buggy_file, rem_start, rem_end, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.finetuning.FineTuningData', 'inference',
        buggy_file, str(rem_start), str(rem_end), tmp_file
    ])

def humaneval_plbart_finetune_input(output_file, humaneval_dir):
    loc_fp = codecs.open(PLBART_FINETUNE_DIR + '../../humaneval/humaneval_loc.txt', 'r', 'utf-8')
    plbart_input = {'config': 'finetune', 'data': {}}
    for line in loc_fp.readlines():
        filename, rem_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = PLBART_FINETUNE_DIR + '../../humaneval/tmp.json'
        get_plbart_finetune_input(humaneval_dir + 'src/main/java/humaneval/buggy/' + filename + '.java', start, end, tmp_file)
        
        if not os.path.exists(tmp_file):
            print(filename, 'failed.', tmp_file, 'not found.')
        print(filename, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        elems = ['buggy function before', 'buggy line', 'buggy function after']
        inputs, outputs = '', ''
        number = 1

        for elem in elems:
            for line in result[elem].split('\n')[:-1]:
                inputs += str(number) + '\t' + re.sub('\\s+', ' ', line).strip()
                if elem == 'buggy line':
                    outputs += str(number) + '\t' + re.sub('\\s+', ' ', line).strip()
                number += 1
        inputs = '<s> ' + inputs + ' </s> java'

        plbart_input['data'][filename] = {
            'loc': rem_loc,
            'input': inputs,
            'gold_output': outputs
        }
        command(['rm', '-rf', tmp_file])
    json.dump(plbart_input, open(output_file, 'w'), indent=2)

def humaneval_plbart_finetune_output(input_file, output_file, model_dir, model_name, index, index, num_output=10):
    tokenizer = PLBartTokenizer.from_pretrained(model_dir + model_name[:-9], src_lang="java", tgt_lang="java")
    model = PLBartForConditionalGeneration.from_pretrained(model_dir + model_name + '/' + index).to(device_ids[0])
    
    plbart_output = json.load(open(input_file, 'r'))
    plbart_output['model'] = model_name
    for i, filename in enumerate(plbart_output['data']):
        text = plbart_output['data'][filename]['input']

        print(i, 'generating', filename)

        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device_ids[0])
        generated_ids = model.generate(
            input_ids, max_length=64, num_beams=10, num_return_sequences=num_output, 
            early_stopping=True, decoder_start_token_id=tokenizer.lang_code_to_id["__java__"]
        )

        output = []
        for generated_id in generated_ids:
            output.append(tokenizer.decode(generated_id, skip_special_tokens=True))
        plbart_output['data'][filename]['output'] = output
    json.dump(plbart_output, open(output_file, 'w'), indent=2)

if __name__ == '__main__':
    model_dir = os.path.abspath('../../models/Java-finetuned-model') + '/'
    model_names = ['plbart-base', 'plbart-large']
    humaneval_dir = PLBART_FINETUNE_DIR + '../../humaneval-java/'
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    device_ids = [0]
    
    import torch
    from transformers import PLBartForConditionalGeneration, PLBartTokenizer
    
    input_dir = PLBART_FINETUNE_DIR + 'humaneval_result/'
    input_file = input_dir + 'plbart_input.json'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(input_file):
        print("==========Preparing input of HumanEval benchmark to finetuned PLBART model==========")
        humaneval_plbart_finetune_input(input_file, humaneval_dir=humaneval_dir)
        print("==========Input written to " + input_file)

    for model_name in model_names:
        for i in range(5):
            output_file = PLBART_FINETUNE_DIR + 'humaneval_result/' + model_name + '_output' + str(i) + '.json'
            print("==========Generating output of HumanEval benchmark by " + model_name + "==========")
            humaneval_plbart_finetune_output(input_file, output_file, model_dir, model_name, str(i), num_output=10)
            print("==========Output written to " + output_file)
