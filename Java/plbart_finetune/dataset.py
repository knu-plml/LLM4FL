import re
import torch
import codecs
import random

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

                    line = re.sub('\\s+', ' ', line).split('//')[0].split('/*')[0]
                    inputs.append(str(number) + '\t' + line)
                    if elem == 'buggy line':
                        outputs.append(str(number) + '\t' + line)
                        faultLocs.append(number)

            noFaultOutputs = ['-1\t there is no fault.']
            interval = max_length // 50
            step = interval // 2
            idx = 0

            # see this at other models.
            # print(currInputs)
            # print(tokenizer.encode(currInputs, add_special_tokens=False, return_tensor='pt'))
            # print(torch.tensor([tokenizer.convert_tokens_to_ids(currInputs)]))

            while True:
                if idx >= len(inputs):
                    break

                currInputs = tokenizer.tokenize('<s> ' + '\n'.join(inputs[idx:idx+interval]) + ' </s> __java__')
                currInputs = torch.tensor([tokenizer.convert_tokens_to_ids(currInputs)])

                currOutputs = []
                for fIdx, f in enumerate(faultLocs):
                    if idx + 1 <= f and f <= idx + 1 + interval:
                        currOutputs.append(outputs[fIdx])

                if currOutputs == []:
                    currOutputs = noFaultOutputs

                currOutputs = tokenizer.tokenize('\n'.join(currOutputs) + ' </s>')
                currOutputs = torch.tensor([tokenizer.convert_tokens_to_ids(currOutputs)])
                currOutputs = torch.cat([torch.LongTensor([[tokenizer.lang_code_to_id["__java__"], 0]]), currOutputs], dim=-1)

                idx += step
                step = interval - step

                if currInputs.size(1) > max_length or currOutputs.size(1) > 32:
                    count += 1
                    continue

                self.data.append({
                    'input_ids': currInputs,
                    'labels': currOutputs,
                    'attention_mask': torch.ones(currInputs.size()).long()
                })

            if (fileNum + 1) % 10000 == 0:
                print('finish loading:', fileNum + 1, 'data size:', len(self.data))
            
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
    max_input_len = max([b['input_ids'].size(1) for b in batch])
    max_output_len = max([b['labels'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_input_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_output_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_input_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data

