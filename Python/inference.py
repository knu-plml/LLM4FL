import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/graphcodebert_finetune/parser'))

import json
import torch
import subprocess
import torch.nn as nn

from graphcodebert_finetune.model import Seq2Seq as graphcodebertSeq2Seq
from graphcodebert_finetune.dataset import convert_strings_to_features
from unixcoder_finetune.model import Seq2Seq as unixcoderSeq2Seq

# GraphCodeBERT, UniXcoder, CodeT5
from transformers import RobertaTokenizer
# CodeGen, InCoder
from transformers import AutoTokenizer


# GraphCodeBERT, UniXcoder
from transformers import RobertaModel, RobertaConfig
# CodeT5
from transformers import T5ForConditionalGeneration
# CodeGen
from transformers import CodeGenForCausalLM
# InCoder
from transformers import AutoModelForCausalLM

def reloadModel(model_name, index, task):
    # Tokenizer
    modelLoc = '../models/' + task + '-finetuned-model/' + model_name + '-finetune/' + index
    if not os.path.exists(modelLoc):
      exit()
    tokenizerLoc = '../models/' + model_name
    model_name = model_name.split('-')[0]
    if model_name == 'graphcodebert' or model_name == 'unixcoder' or model_name == 'codet5':
        tokenizer = RobertaTokenizer.from_pretrained(tokenizerLoc)
    elif model_name == 'codegen' or model_name == 'incoder':
        tokenizer = AutoTokenizer.from_pretrained(tokenizerLoc)

    # Model
    if model_name == 'graphcodebert':
        encoder = RobertaModel.from_pretrained(tokenizerLoc)
        config = RobertaConfig.from_pretrained(tokenizerLoc)
        decoderLayer = nn.TransformerDecoderLayer(d_model=768, nhead=12)
        decoder = nn.TransformerDecoder(decoderLayer, num_layers=6)
        model = graphcodebertSeq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=10, max_length=32, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        model.load_state_dict(torch.load(modelLoc + '/model.bin', map_location="cuda:" + str(device_ids[0])))
    elif model_name == 'unixcoder':
        config = RobertaConfig.from_pretrained(tokenizerLoc)
        config.is_decoder = True
        encoder = RobertaModel.from_pretrained(tokenizerLoc, config=config)
        model = unixcoderSeq2Seq(encoder=encoder, decoder=encoder, config=config,
                        beam_size=10, max_length=32,
                        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"][0]),
                        eos_id=tokenizer.sep_token_id)
        model.load_state_dict(torch.load(modelLoc + '/model.bin', map_location="cuda:" + str(device_ids[0])))
    elif model_name == 'codet5':
        model = T5ForConditionalGeneration.from_pretrained(modelLoc)
    elif model_name == 'codegen':
        model = CodeGenForCausalLM.from_pretrained(modelLoc)
    elif model_name == 'incoder':
        if len(device_ids) > 1:
            model = AutoModelForCausalLM.from_pretrained(modelLoc)
        else:
            model = AutoModelForCausalLM.from_pretrained(modelLoc, device_map="auto")

    if len(device_ids) == 1:
        model = model.to(device_ids[0])

    return tokenizer, model

def python_codet5_finetune_input(output_file, maxLen):
    data = json.load(open('../data/codeNetPython.jsonl', 'r'))
    result = {}
    for d in data:
        file = d['src_id']
        print(file)
        faultCode = d['src']
        correctCode = d['tgt']

        inputs, outputs = [], []
        number = 1
        count = 0
        start, end = int(120000 * 0.9), 120000

        for s, t in zip(faultCode.strip().split('\n'), correctCode.strip().split('\n')):
            s = s.strip()
            t = t.strip()

            if s == '':
                continue
            inputs.append(str(number) + '\t' + s)
            if s != t:
                if number == 1:
                    break
                outputs.append(str(number) + '\t' + s)
            number += 1
        if number == 1:
            continue

        if len(outputs) > 1:
            continue

        count += 1
        if count >= start and count < end:
            noFaultOutputs = ['-1\t there is no fault.']
            interval = (int(maxLen) - 32) // 30 # avg., there are 30 tokens at one line.
            step = interval // 2

            idx = 0
            seqIdx = 0

            result[file] = {}
            result[file]['input'] = '\n'.join(inputs)
            result[file]['output'] = '\n'.join(outputs)

            while True:
                if idx >= len(inputs):
                    break

                currInputs = inputs[idx:idx+interval]
                currOutputs = []
                
                for output in outputs:
                    for currInput in currInputs:
                        if output == currInput:
                            currOutputs.append(output)

                idx += step
                step = interval - step
                seqIdx += 1
                    
                if currOutputs == []:
                    currOutputs = noFaultOutputs
                    
                result[file]['seq' + str(seqIdx)] = {
                    'code': '\n'.join(currInputs),
                    'seqs': '\n'.join(currOutputs),
                }
        elif count > end:
            break
    json.dump(result, open(output_file, 'w'), indent=2)

def codeToIds(model_name, code, tokenizer, maxLen):
    others = []
    if model_name == 'graphcodebert': 
        codeIds, sMask, posIdx, mask, tIds, tMask = convert_strings_to_features(code, '', tokenizer, 512)
        others = [sMask, posIdx, mask, tIds, tMask]
    elif model_name == 'unixcoder':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.bos_token] + ["<encoder-decoder>"] + [tokenizer.sep_token]) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-5] + tokenizer.convert_tokens_to_ids(["<mask0>"] + [tokenizer.sep_token])])
    elif model_name == 'codet5':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.bos_token]) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-2] + tokenizer.convert_tokens_to_ids([tokenizer.sep_token])])
    elif model_name == 'codegen':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))])
    elif model_name == 'incoder':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<|endoftext|>')) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-1]])

    if type(codeIds) is list:
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids(codeIds)])

    return codeIds, others

def generateOutput(model, tokenizer, model_name, code, others):
    code = code.to(device_ids[0])
    generated_output = False
    if model_name == 'graphcodebert': 
        scores, preds = model(code, others[0].to(device_ids[0]), others[1].to(device_ids[0]), others[2].to(device_ids[0]))
    elif model_name == 'unixcoder':
        scores, preds = model(code)
    elif model_name == 'codet5':
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        generated_output = model.generate(code, max_new_tokens=32, num_beams=10, num_return_sequences=10, pad_token_id=eos_id, eos_token_id=eos_id, early_stopping=True, return_dict_in_generate=True, output_scores=True)
    elif model_name == 'codegen':
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        generated_output = model.generate(code, max_new_tokens=32, num_beams=10, num_return_sequences=10, pad_token_id=eos_id, eos_token_id=eos_id, early_stopping=True, return_dict_in_generate=True, output_scores=True)
    elif model_name == 'incoder':
        eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')
        generated_output = model.generate(code, max_new_tokens=32, num_beams=10, num_return_sequences=10, pad_token_id=eos_id, eos_token_id=eos_id, early_stopping=True, return_dict_in_generate=True, output_scores=True)

    sequences = []

    if generated_output:
        scores = generated_output.sequences_scores.tolist()
        for output in generated_output.sequences:
            sequences.append(tokenizer.decode(output, skip_special_tokens=True))
    else:
        scores = scores.tolist()
        for output in preds[0]:
            sequences.append(tokenizer.decode(output, skip_special_tokens=True))

    return scores, sequences

def python_codet5_finetune_output(input_file, output_file, task, model_name, index, maxLen, num_output=10):
    tokenizer, model = reloadModel(model_name, index, task)
    
    data = json.load(open(input_file, 'r'))
    for file in data.keys():
        print(file)
        idx = 1
        while 'seq' + str(idx) in data[proj][bugId][file]:
            currSeqNum = 'seq' + str(idx)
            #print(proj, bugId, file, currSeqNum)

            output = []
            currCode = data[file][currSeqNum]['code']
            outputSeq = data[file][currSeqNum]['seqs']

            inputSeq, others = codeToIds(model_name.split('-')[0], currCode, tokenizer, int(maxLen)-32)

            scores, outputs = generateOutput(model, tokenizer, model_name.split('-')[0], inputSeq, others)

            data[file][currSeqNum]['score'] = scores
            data[file][currSeqNum]['output'] = outputs
            idx += 1

    json.dump(data, open(output_file, 'w'), indent=2)

if __name__ == '__main__':
    maxLen = sys.argv[1] # 512, 1024
    task = sys.argv[2] # Java, ...
    model_name = sys.argv[3] # codebert-base, ...
    device_ids = list(map(int, sys.argv[4].split(','))) # 1,2 ...
    iterN = int(sys.argv[5])

    input_dir = '../data/'
    input_file = '../data/' + 'python_' + task + '_' + str(maxLen) + '.json'
    if not os.path.exists(input_file):
        print("==========Preparing input of Defects4J benchmark==========")
        python_codet5_finetune_input(input_file, maxLen)
        print("==========Input written to " + input_file + "==========")

    output_dir = './' + model_name.split('-')[0] + '_finetune/python/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = './' + model_name.split('-')[0] + '_finetune/python/' + model_name + '_result' + str(iterN) + '.json'
    print("==========Generating output of python benchmark by " + model_name + "==========")
    python_codet5_finetune_output(input_file, output_file, task, model_name, str(iterN), maxLen, num_output=10)
    print("==========Output written to " + output_file)
