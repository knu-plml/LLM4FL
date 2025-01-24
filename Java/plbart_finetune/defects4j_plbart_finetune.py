import re
import os
import sys
import json
import codecs
import numpy as np
import subprocess
PROJECT = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']

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

def defects4j_plbart_finetune_input(output_file, tmp_dir):
    loc_fp = json.load(codecs.open(PLBART_FINETUNE_DIR + '../../defects4j/defects4j_loc.json', 'r'))
    plbart_input = {'config': 'finetune', 'data': {}}
    for proj in PROJECT:
        KeyList = list(loc_fp[proj].keys())
        for bug_id in KeyList:
            funList = list(loc_fp[proj][bug_id].keys())
            for func in funList:
                path = loc_fp[proj][bug_id][func]['source']
                rem_loc = loc_fp[proj][bug_id][func]['fileLocation']
                gold_outputs = loc_fp[proj][bug_id][func]['functionLocation']
                start, end = rem_loc.split(' ')[0].split('-')
                tmp_file = PLBART_FINETUNE_DIR + '../../defects4j/plbart_tmp.json'

                subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                get_plbart_finetune_input(tmp_dir + path, start, end, tmp_file)
                
                if not os.path.exists(tmp_file):
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                    continue
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))

                if result["buggy function before"].strip() == '' and result["buggy line"].strip() == '' and result["buggy function after"].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')
                    continue

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

                plbart_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
                    'loc': rem_loc,
                    'function': func,
                    'input': inputs,
                    'gold_output': gold_outputs
                }
                command(['rm', '-rf', tmp_file])
                command(['rm', '-rf', tmp_dir])
        
    json.dump(plbart_input, open(output_file, 'w'), indent=2)

def defects4j_plbart_finetune_output(input_file, output_file, model_dir, model_name, index, num_output=10):
    tokenizer = PLBartTokenizer.from_pretrained(model_dir + model_name[:-9], src_lang="java", tgt_lang="java")
    model = PLBartForConditionalGeneration.from_pretrained(model_dir + model_name + '/' + index).to(device_ids[0])

    plbart_output = json.load(open(input_file, 'r'))
    plbart_output['model'] = model_name
    for i, filename in enumerate(plbart_output['data']):
        text = plbart_output['data'][filename]['input']

        print(i + 1, 'generating', filename)

        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device_ids[0])

        if input_ids.size(1) >= 512:
            print('too long:', input_ids.size(1))
            continue

        generated_ids = model.generate(
            input_ids, max_length=128, num_beams=num_output, num_return_sequences=num_output, 
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
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
    device_ids = [0]
    
    import torch
    from transformers import PLBartForConditionalGeneration, PLBartTokenizer

    input_dir = PLBART_FINETUNE_DIR + 'defects4j_result/'
    input_file = input_dir + 'plbart_input.json'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(input_file):
        print("==========Preparing input of Defects4J benchmark to finetuned PLBART model==========")
        defects4j_plbart_finetune_input(input_file, tmp_dir='/tmp/plbart/')
        print("==========Input written to " + input_file)
   
    for model_name in model_names:
        for i in range(5):
            output_file = PLBART_FINETUNE_DIR + 'defects4j_result/' + model_name + '_output' + str(i) + '.json'
            print("==========Generating output of Defects4J benchmark by " + model_name + "==========")
            defects4j_plbart_finetune_output(input_file, output_file, model_dir, model_name, str(i), num_output=10)
            print("==========Output written to " + output_file)
