import codecs
import os
import numpy as np
import sys
import json
import subprocess

INCODER_FINETUNE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = INCODER_FINETUNE_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_incoder_finetune_input(buggy_file, rem_start, rem_end, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.finetuning.FineTuningData', 'inference',
        buggy_file, str(rem_start), str(rem_end), tmp_file
    ])

def quixbugs_incoder_finetune_input(output_file):
    loc_fp = codecs.open(INCODER_FINETUNE_DIR + '../../quixbugs/quixbugs_loc.txt', 'r', 'utf-8')
    incoder_input = {'config': 'finetune', 'data': {}}
    for line in loc_fp.readlines():
        filename, rem_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = INCODER_FINETUNE_DIR + '../../quixbugs/tmp.json'
        get_incoder_finetune_input(INCODER_FINETUNE_DIR + '../../quixbugs/QuixBugs/changed_java_programs/' + filename + '.java', start, end, tmp_file)
        
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

        incoder_input['data'][filename] = {
            'loc': rem_loc,
            'input': inputs,
            'gold_output': outputs
        }
        command(['rm', '-rf', tmp_file])
    json.dump(incoder_input, open(output_file, 'w'), indent=2)


def quixbugs_incoder_finetune_output(input_file, output_file, model_dir, model_name, index, num_output=10):
    tokenizer = AutoTokenizer.from_pretrained(model_dir + model_name[:-9])
    if len(device_ids) > 1:
        model = AutoModelForCausalLM.from_pretrained(model_dir + model_name + '/' + index, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir + model_name + '/' + index).to(device_ids[0])
    
    incoder_output = json.load(open(input_file, 'r'))
    incoder_output['model'] = model_name
    for i, filename in enumerate(incoder_output['data']):
        text = incoder_output['data'][filename]['input']

        print(i + 1, 'generating', filename)
        
        try:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device_ids[0])
            if input_ids.size(1) >= 1024:
                print('too long:', input_ids.size(1))
                continue

            eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')
            generated_ids = model.generate(
                input_ids, max_new_tokens=128, num_beams=num_output, num_return_sequences=num_output, early_stopping=True,
                pad_token_id=eos_id, eos_token_id=eos_id
            )
        except Exception as e:
            print(e)
            continue

        output = []
        for generated_id in generated_ids:
            output.append(tokenizer.decode(generated_id, skip_special_tokens=False))
        incoder_output['data'][filename]['output'] = output
    json.dump(incoder_output, open(output_file, 'w'), indent=2)

if __name__ == '__main__':
    model_dir = os.path.abspath('../../models/Java-finetuned-model') + '/'
    model_names = ['incoder-1B', 'incoder-6B']
    iterN = sys.argv[1]
    
    input_dir = INCODER_FINETUNE_DIR + 'quixbugs_result/'
    input_file = input_dir + 'incoder_input.json'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(input_file):
        print("==========Preparing input of QuixBugs benchmark to finetuned INCODER model==========")
        quixbugs_incoder_finetune_input(input_file)
        print("==========Input written to " + input_file)
    
    for model_name in model_names:
        if model_name == 'incoder-1B-finetune':
            os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES']="0, 1, 2"
            device_ids = [0]
        elif model_name == 'incoder-6B-finetune':
            device_ids = [0, 1, 2, 3]

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        output_file = INCODER_FINETUNE_DIR + 'quixbugs_result/' + model_name + '_output' + iterN + '.json'

        print("==========Generating output of QuixBugs benchmark by " + model_name + "==========")
        quixbugs_incoder_finetune_output(input_file, output_file, model_dir, model_name, iterN, num_output=10)
        print("==========Output written to " + output_file)
