import torch
import json 
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024, shuffle=False, load_range=None):
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

            noFaultOutputs = ['-1\t there is no fault.']
            interval = max_length // 50 # avg., there are 30 tokens at one line.
            step = interval // 2
            idx = 0

            while True:
                if idx >= len(inputs):
                    break

                currInputs = tokenizer.tokenize('<|endoftext|>' + '\n'.join(inputs[idx:idx+interval])) # + [tokenizer.eos_token]

                currOutputs = []
                for fIdx, f in enumerate(outputs):
                    faultIdx = int(f.split('\t')[0])
                    if idx + 1 <= faultIdx and faultIdx <= idx + 1 + interval:
                        currOutputs.append(f)

                if currOutputs == []:
                    currOutputs = noFaultOutputs

                currOutputs = tokenizer.tokenize('\n'.join(currOutputs) + '<|endofmask|>')

                currInputs += currOutputs

                currInputs = torch.tensor([tokenizer.convert_tokens_to_ids(currInputs)])
                currOutputs = torch.tensor([[2] + tokenizer.convert_tokens_to_ids(currOutputs)])

                idx += step
                step = interval - step

                if currInputs.size(1) > max_length + 32:
                    count += 1
                    continue

                self.data.append({
                    'input_ids': currInputs,
                    'labels': torch.cat([torch.zeros(1, currInputs.size(1) - currOutputs.size(1)).fill_(-100).long(), currOutputs], dim=1),
                    'attention_mask': torch.ones(currInputs.size()).long()
                })

            if (fileNum + 1) % 10000 == 0:
                print('finish loading:', fileNum + 1, 'data size', len(self.data))
            
            if load_range is not None and fileNum+1 == load_range[1]:
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
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_len = max([b['input_ids'].size(1) for b in batch])
    eos_id = 50517
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_len - b['input_ids'].size(1)).fill_(eos_id).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data

