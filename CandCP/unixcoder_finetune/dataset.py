import torch
import json
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, shuffle=False, load_range=None):
        self.data = []
        self.max_length = max_length

        dataList = json.load(open(file_path))
        count = 0
        for fileNum, l in enumerate(dataList):
            elem = ['code', 'code_after']
            inputs, outputs = [], []
            number = 1

            for s, t in zip(l[elem[0]].strip().split('\n'), l[elem[1]].strip().split('\n')):
                s = s.strip()
                t = t.strip()
                if s == '' or t == '':
                    continue
                inputs.append(str(number) + '\t' + s)
                if s != t:
                    if number == 1:
                        continue
                    outputs.append(str(number) + '\t' + s)
                number += 1

            if number == 1:
                continue

            #print(inputs)
            #print(tokenizer.encode('\n'.join(inputs)))
            #print()
            #print(tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + ["<encoder-decoder>"] + [tokenizer.sep_token] + tokenizer.tokenize('\n'.join(inputs)) + ["<mask0>"] + [tokenizer.sep_token]))
            #print()
            #print(tokenizer.convert_tokens_to_ids([tokenizer.bos_token] + ["<encoder-decoder>"] + [tokenizer.sep_token] + tokenizer.tokenize('\n'.join(inputs)) + ["<mask0>"] + [tokenizer.sep_token]))
            #exit()

            noFaultOutputs = ['-1\t there is no fault.']
            interval = max_length // 50 # avg., there are 30 tokens at one line.
            step = interval // 2
            idx = 0

            while True: 
                if idx >= len(inputs):
                    break

                currInputs = [tokenizer.bos_token] + ["<encoder-decoder>"] + [tokenizer.sep_token] + tokenizer.tokenize('\n'.join(inputs[idx:idx+interval])) + ["<mask0>"] + [tokenizer.sep_token]
                currInputs = torch.tensor([tokenizer.convert_tokens_to_ids(currInputs)])
                
                currOutputs = []
                for fIdx, f in enumerate(outputs):
                    faultIdx = int(f.split('\t')[0])
                    if idx + 1 <= faultIdx and faultIdx <= idx + 1 + interval:
                        currOutputs.append(f)

                if currOutputs == []:
                    currOutputs = noFaultOutputs

                currOutputs = ["<mask0>"] + tokenizer.tokenize('\n'.join(currOutputs)) + [tokenizer.sep_token]
                currOutputs = torch.tensor([tokenizer.convert_tokens_to_ids(currOutputs)])

                idx += step
                step = interval - step

                if currInputs.size(1) > max_length + 5 or currOutputs.size(1) > 32:
                    count += 1
                    continue

                self.data.append({
                    'input_ids': currInputs,
                    'labels': currOutputs,
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
    batch_data = {'input_ids': [], 'labels': []}
    max_input_len =  max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).long()], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    return batch_data

