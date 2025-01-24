import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import codecs
import random
import numpy as np

from parser import DFG_java
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function={
    'java':DFG_java
    #'python':DFG_python
}

parsers={}

for lang in dfg_function:
    #LANGUAGE = Language('parser/my-languages.so', lang)
    LANGUAGE = Language('graphcodebert_finetune/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser,dfg_function[lang]]
    parsers[lang]= parser

#remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
#remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    try:
        tree = parser[0].parse(bytes(code,'utf8'))
        root_node = tree.root_node
        tokens_index=tree_to_token_index(root_node)
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{})
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

def convert_strings_to_features(source, target, tokenizer, maxLength):
    features = []
    code_tokens,dfg=extract_dataflow(source, parsers['java'], 'java')
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))
    code_tokens=[y for x in code_tokens for y in x]
    code_tokens=code_tokens[:maxLength-3]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg=dfg[:maxLength-len(source_tokens)]
    source_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    source_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=maxLength-len(source_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask = [1] * (len(source_tokens))
    source_mask+=[0]*padding_length

    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)

    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]

    noFaultOutputs = '-1\t there is no fault.'

    if target == '':
        target_tokens = tokenizer.tokenize(noFaultOutputs)
    else:
        target_tokens = tokenizer.tokenize(target)[:30]
    target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] *len(target_ids)
    padding_length = 32 - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length
    target_mask+=[0]*padding_length

    mask=np.zeros((maxLength,maxLength),dtype=np.bool_)
    node_index=sum([i>1 for i in position_idx])
    max_length=sum([i!=1 for i in position_idx])
    mask[:node_index,:node_index]=True

    for idx,i in enumerate(source_ids):
        if i in [0,2]:
            mask[idx,:max_length]=True

    for idx,(a,b) in enumerate(dfg_to_code):
        if a<node_index and b<node_index:
            mask[idx+node_index,a:b]=True
            mask[a:b,idx+node_index]=True

    for idx,nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a+node_index<len(position_idx):
                mask[idx+node_index,a+node_index]=True

    return (torch.tensor([source_ids]),
            torch.tensor([source_mask]),
            torch.tensor([position_idx]),
            torch.tensor(np.array([mask])),
            torch.tensor([target_ids]),
            torch.tensor([target_mask]))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, shuffle=False, load_range=None):
        self.data = []
        self.max_length = max_length

        fp = codecs.open(file_path, 'r', 'utf-8')
        count = 0
        for fileNum, l in enumerate(fp.readlines()):
            l = eval(l)

            elems = ['buggy function before', 'buggy line', 'buggy function after']
            inputs, outputs, faultLocs = [], [], []
            number = 0

            for elem in elems:
                for line in l[elem].split('\n')[:-1]:
                    if line.strip() == '' or (line.strip()[:2] == '//' or line.strip()[:2] == '/*' or line.strip()[0] == '*' or line.strip()[-2:] == '*/'):
                        continue
                    elif line.strip().replace(')', '').replace('}', '').replace(';', '').replace('{', '') == '':
                        if len(inputs) >= 1:
                            inputs[-1] = inputs[-1] + ' ' + line.strip()
                            if len(outputs) >= 1:
                                outputs[-1] = outputs[-1] + ' ' + line.strip()
                        continue
                    else:
                        number += 1

                    line = line.split('//')[0].split('/*')[0]
                    inputs.append(str(number) + '\t' + line)
                    if elem == 'buggy line':
                        outputs.append(str(number) + '\t' + line)
                        faultLocs.append(number)

            interval = max_length // 50 # avg., there are 30 tokens at one line.
            step = interval // 2
            idx = 0

            while True: 
                if idx >= len(inputs):
                    break
                
                source = '\n'.join(inputs[idx:interval])
                
                currOutputs = []
                for fIdx, f in enumerate(faultLocs):
                    if idx + 1 <= f and f <= idx + 1 + interval:
                        currOutputs.append(outputs[fIdx])


                target = '\n'.join(currOutputs)

                currInputs, inputMask, posIdx, mask, currOutputs, outputMask = convert_strings_to_features(source, target, tokenizer, max_length)

                idx += step
                step = interval - step

                if currInputs.size(1) > max_length + 2 or currOutputs.size(1) > 32:
                    count += 1
                    continue

                self.data.append({
                    'input_ids': currInputs,
                    'labels': currOutputs,
                    'pos_idx': posIdx,
                    'attention_mask': mask,
                    'input_mask': inputMask,
                    'labels_mask': outputMask
                })

            if (fileNum + 1) % 10000 == 0:
                print('finish loading:', fileNum + 1, 'data size:', len(self.data))
            
            if load_range is not None and fileNum+1 >= load_range[1]:
                break
        print('long inputs:', count)
        
        if shuffle:
            random.shuffle(self.data)

        print(file_path, 'total size:', len(self.data))

        if load_range is not None:
            self.data = self.data[load_range[0]: ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]


def custom_collate(batch):
    batch_data = {'input_ids': [], 'labels': [], 'pos_idx': [], 'attention_mask': [], 'input_mask': [], 'labels_mask': []}
    for b in batch:
        batch_data['input_ids'].append(b['input_ids'])
        batch_data['labels'].append(b['labels'])
        batch_data['pos_idx'].append(b['pos_idx'])
        batch_data['attention_mask'].append(b['attention_mask'])
        batch_data['input_mask'].append(b['input_mask'])
        batch_data['labels_mask'].append(b['labels_mask'])
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['pos_idx'] = torch.cat(batch_data['pos_idx'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    batch_data['input_mask'] = torch.cat(batch_data['input_mask'], dim=0)
    batch_data['labels_mask'] = torch.cat(batch_data['labels_mask'], dim=0)
    return batch_data

