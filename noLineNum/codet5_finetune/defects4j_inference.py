import codecs
import os
import numpy as np
import sys
import json
import subprocess
PROJECT = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']

CODET5_FINETUNE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = CODET5_FINETUNE_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_codet5_finetune_input(buggy_file, rem_start, rem_end, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.finetuning.FineTuningData', 'inference',
        buggy_file, str(rem_start), str(rem_end), tmp_file
    ])

def codet5_finetune_input(output_file, tmp_dir):
    loc_fp = json.load(codecs.open(CODET5_FINETUNE_DIR + '../../defects4j/defects4j_loc.json', 'r'))
    codet5_input = {'config': 'finetune', 'data': {}}
    for proj in PROJECT:
        KeyList = list(loc_fp[proj].keys())
        for bug_id in KeyList:
            funList = list(loc_fp[proj][bug_id].keys())
            for func in funList:
                path = loc_fp[proj][bug_id][func]['source']
                rem_loc = loc_fp[proj][bug_id][func]['fileLocation']
                gold_outputs = loc_fp[proj][bug_id][func]['functionLocation']
                start, end = rem_loc.split(' ')[0].split('-')
                tmp_file = CODET5_FINETUNE_DIR + '../../defects4j/codet5_tmp.json'

                subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                get_codet5_finetune_input(tmp_dir + path, start, end, tmp_file)
                
                if not os.path.exists(tmp_file):
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                    continue
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))

                if result["buggy function before"].strip() == '' and result["buggy line"].strip() == '' and result["buggy function after"].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')
                    continue

                elems = ['buggy function before', 'buggy line', 'buggy function after']
                inputs= ''

                for elem in elems:
                    for line in result[elem].split('\n')[:-1]:
                        inputs += line + '\n'

                codet5_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
                    'loc': rem_loc,
                    'function': func,
                    'input': inputs,
                    'gold_output': result['buggy line']
                }
                command(['rm', '-rf', tmp_file])
                command(['rm', '-rf', tmp_dir])
    json.dump(codet5_input, open(CODET5_FINETUNE_DIR + output_file, 'w'), indent=2)


def codet5_finetune_output(input_file, output_file, model_dir, model_name, iterN, num_output=10):
    tokenizer = RobertaTokenizer.from_pretrained(CODET5_FINETUNE_DIR + '/'.join(model_dir.split('/')[:-2]) + '/' + model_name[:-9])
    model = T5ForConditionalGeneration.from_pretrained(CODET5_FINETUNE_DIR + model_dir + model_name + '/' + iterN).to(device_ids[0])

    codet5_output = json.load(open(CODET5_FINETUNE_DIR + input_file, 'r'))
    codet5_output['model'] = model_name
    for i, filename in enumerate(codet5_output['data']):
        text = codet5_output['data'][filename]['input']

        print(i + 1, 'generating', filename)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device_ids[0])
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        if input_ids.size(1) >= 512:
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
            output.append(tokenizer.decode(generated_id, skip_special_tokens=True))
        codet5_output['data'][filename]['output'] = output
    json.dump(codet5_output, open(CODET5_FINETUNE_DIR + output_file, 'w'), indent=2)

if __name__ == '__main__':
    model_dir = '../../models/noLineNum-finetuned-model/'

    model_names = ['codet5-small-finetune', 'codet5-base-finetune', 'codet5-large-finetune']
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="3"
    device_ids = [0]

    import torch
    from transformers import RobertaTokenizer, T5ForConditionalGeneration

    
    input_dir = './defects4j_result/'
    input_file = input_dir + 'codet5_input.json'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(input_file):
        print("==========Preparing input of benchmark to finetuned codet5 model==========")
        codet5_finetune_input(input_file, tmp_dir='/tmp/codet5/')
        print("==========Input written to " + input_file)

    for model_name in model_names:
        for i in range(5):
            output_file = 'defects4j_result/' + model_name + '_output' + str(i) + '.json'
            print("==========Generating output of benchmark by " + model_name + "==========")
            codet5_finetune_output(input_file, output_file, model_dir, model_name, str(i), num_output=10)
            print("==========Output written to " + output_file)
