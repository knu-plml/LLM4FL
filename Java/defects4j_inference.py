import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/graphcodebert_finetune/parser'))
import json
import torch
import subprocess
import torch.nn as nn
#import numpy as np

from codebert_finetune.model import Seq2Seq as codebertSeq2Seq
from graphcodebert_finetune.model import Seq2Seq as graphcodebertSeq2Seq
from graphcodebert_finetune.dataset import convert_strings_to_features
from unixcoder_finetune.model import Seq2Seq as unixcoderSeq2Seq

# CodeBERT, GraphCodeBERT, UniXcoder, CodeT5
from transformers import RobertaTokenizer
# PLBART
from transformers import PLBartTokenizer 
# CodeGen, InCoder
from transformers import AutoTokenizer


# CodeBERT, GraphCodeBERT, UniXcoder
from transformers import RobertaModel, RobertaConfig
# CodeT5
from transformers import T5ForConditionalGeneration
# PLBART
from transformers import PLBartForConditionalGeneration
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
    if model_name == 'codebert' or model_name == 'graphcodebert' or model_name == 'unixcoder' or model_name == 'codet5':
        tokenizer = RobertaTokenizer.from_pretrained(tokenizerLoc)
    elif model_name == 'plbart':
        tokenizer = PLBartTokenizer.from_pretrained(tokenizerLoc, src_lang='java', tgt_lang='java')
    elif model_name == 'codegen' or model_name == 'incoder':
        tokenizer = AutoTokenizer.from_pretrained(tokenizerLoc)

    # Model
    if model_name == 'codebert' or model_name == 'graphcodebert':
        encoder = RobertaModel.from_pretrained(tokenizerLoc)
        config = RobertaConfig.from_pretrained(tokenizerLoc)
        decoderLayer = nn.TransformerDecoderLayer(d_model=768, nhead=12)
        decoder = nn.TransformerDecoder(decoderLayer, num_layers=6)
        if model_name == 'codebert':
            model = codebertSeq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=10, max_length=32, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        elif model_name == 'graphcodebert':
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
    elif model_name == 'plbart':
        model = PLBartForConditionalGeneration.from_pretrained(modelLoc)
    elif model_name == 'codegen':
        if len(device_ids) > 1:
            model = CodeGenForCausalLM.from_pretrained(modelLoc, device_map="auto")
        else:
            model = CodeGenForCausalLM.from_pretrained(modelLoc)
    elif model_name == 'incoder':
        if len(device_ids) > 1:
            model = AutoModelForCausalLM.from_pretrained(modelLoc)
        else:
            model = AutoModelForCausalLM.from_pretrained(modelLoc, device_map="auto")

    if len(device_ids) == 1:
        #model = nn.DataParallel(model, device_ids=[device_ids[0]]).to(device_ids[0])
        model = model.to(device_ids[0])

    return tokenizer, model

def defects4j_codet5_finetune_input(output_file, maxLen):
    data = json.load(open('../data/codeAndCodeWLineforGPT.json', 'r'))
    for proj in data.keys():
        for bugId in data[proj].keys():
            for file in data[proj][bugId].keys():
                print(proj, bugId, file)
                code = data[proj][bugId][file]['codeWLine']
                locs = data[proj][bugId][file]['newFaultLocs']
                seqs = data[proj][bugId][file]['newFaultSeqs']

                noFaultOutputs = ['-1\t there is no fault.']
                interval = (int(maxLen) - 32) // 30 # avg., there are 30 tokens at one line.
                step = interval // 2

                idx = 0
                seqIdx = 0

                inputs = code.split('\n')[:-1]

                while True:
                    if idx >= len(inputs):
                        break

                    currInputs = inputs[idx:idx+interval]
                    currOutputs = []
                    
                    if locs.strip() != '':
                        for f in locs.split(','):
                            currStart, currEnd = map(int, f.split('~'))
                            for currIdx in (currStart, currEnd+1):
                                if idx + 1 <= currIdx and currIdx <= idx + 1 + interval:
                                    for s in seqs.split('\n')[:-1]:
                                        if int(s.split('\t')[0]) == currIdx:
                                            currOutputs.append(s)

                    idx += step
                    step = interval - step
                    seqIdx += 1
                        
                    if currOutputs == []:
                        currOutputs = noFaultOutputs
                        
                    data[proj][bugId][file]['seq' + str(seqIdx)] = {
                        'code': '\n'.join(currInputs),
                        'seqs': '\n'.join(currOutputs),
                    }
    json.dump(data, open(output_file, 'w'), indent=2)

def codeToIds(model_name, code, tokenizer, maxLen):
    others = []
    if model_name == 'codebert':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.bos_token]) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-2] + tokenizer.convert_tokens_to_ids([tokenizer.sep_token])])
        others = [torch.ones(codeIds.size()).long()]
    elif model_name == 'graphcodebert': 
        codeIds, sMask, posIdx, mask, tIds, tMask = convert_strings_to_features(code, '', tokenizer, 512)
        others = [sMask, posIdx, mask, tIds, tMask]
    elif model_name == 'unixcoder':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.bos_token] + ["<encoder-decoder>"] + [tokenizer.sep_token]) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-5] + tokenizer.convert_tokens_to_ids(["<mask0>"] + [tokenizer.sep_token])])
    elif model_name == 'codet5':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.bos_token]) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-2] + tokenizer.convert_tokens_to_ids([tokenizer.sep_token])])
    elif model_name == 'plbart':
        codeIds = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<s> ')) + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))[:maxLen-3] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' </s> __java__'))])
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
    if model_name == 'codebert':
        scores, preds = model(source_ids = code, source_mask=others[0].to(device_ids[0]))
    elif model_name == 'graphcodebert': 
        scores, preds = model(code, others[0].to(device_ids[0]), others[1].to(device_ids[0]), others[2].to(device_ids[0]))
    elif model_name == 'unixcoder':
        scores, preds = model(code)
    elif model_name == 'codet5':
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        generated_output = model.generate(code, max_new_tokens=32, num_beams=10, num_return_sequences=10, pad_token_id=eos_id, eos_token_id=eos_id, early_stopping=True, return_dict_in_generate=True, output_scores=True)
    elif model_name == 'plbart':
        decoder_start_id = tokenizer.lang_code_to_id["__java__"]
        generated_output = model.generate(code, max_length=32, num_beams=10, num_return_sequences=10, early_stopping=True, decoder_start_token_id=decoder_start_id, return_dict_in_generate=True, output_scores=True)
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

def defects4j_codet5_finetune_output(input_file, output_file, task, model_name, index, maxLen, num_output=10):
    tokenizer, model = reloadModel(model_name, index, task)
    
    data = json.load(open(input_file, 'r'))
    for proj in data.keys():
        print(proj)
        for bugId in data[proj].keys():
            print(proj, bugId)
            for file in data[proj][bugId].keys():
                print(proj, bugId, file)
                idx = 1
                while 'seq' + str(idx) in data[proj][bugId][file]:
                    currSeqNum = 'seq' + str(idx)
                    #print(proj, bugId, file, currSeqNum)

                    output = []
                    currCode = data[proj][bugId][file][currSeqNum]['code']
                    outputSeq = data[proj][bugId][file][currSeqNum]['seqs']

                    inputSeq, others = codeToIds(model_name.split('-')[0], currCode, tokenizer, int(maxLen)-32)

                    scores, outputs = generateOutput(model, tokenizer, model_name.split('-')[0], inputSeq, others)

                    data[proj][bugId][file][currSeqNum]['score'] = scores
                    data[proj][bugId][file][currSeqNum]['output'] = outputs
                    idx += 1

    json.dump(data, open(output_file, 'w'), indent=2)

if __name__ == '__main__':
    maxLen = sys.argv[1] # 512, 1024
    task = sys.argv[2] # Java, ...
    model_name = sys.argv[3] # codebert-base, ...
    device_ids = list(map(int, sys.argv[4].split(','))) # 1,2 ...
    iterN = int(sys.argv[5])

    input_dir = '../data/'
    input_file = '../data/' + 'defects4j_' + task + '_' + str(maxLen) + '.json'
    if not os.path.exists(input_file):
        print("==========Preparing input of Defects4J benchmark==========")
        defects4j_codet5_finetune_input(input_file, maxLen)
        print("==========Input written to " + input_file + "==========")

    output_dir = './' + model_name.split('-')[0] + '_finetune/defects4j/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = './' + model_name.split('-')[0] + '_finetune/defects4j/' + model_name + '_result' + str(iterN) + '.json'
    print("==========Generating output of Defects4J benchmark by " + model_name + "==========")
    defects4j_codet5_finetune_output(input_file, output_file, task, model_name, str(iterN), maxLen, num_output=10)
    print("==========Output written to " + output_file)
