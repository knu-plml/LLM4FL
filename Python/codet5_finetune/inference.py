import os
import sys
import json

def codet5_finetune_input(output_file, data_dir):
  dataList = json.load(open(data_dir))
  new_data = []
  for data in dataList:
    elems = ['src', 'tgt']
    inputs, outputs = '', ''
    number = 1

    for s, t in zip(data[elems[0]].strip().split('\n'), data[elems[1]].strip().split('\n')):
      s = s.strip()
      t = t.strip()
      if s == '':
        continue
      inputs += str(number) + '\t' + s + '\n'
      if s != t:
        if number == 1:
          continue
        outputs += str(number) + '\t' + s + '\n'
      number += 1
    if number == 1:
      continue
    if len(outputs.split('\n')) != 2:
      continue
    n_data = {}
    n_data['inputs'] = inputs
    n_data['outputs'] = outputs
    new_data.append(n_data)

  new_data = new_data[int(120000 * 0.9):120000]
  codet5_input = {'config': 'finetune', 'data': {}}
  for idx, data in enumerate(new_data):
    inputs, outputs = data['inputs'], data['outputs']
    codet5_input['data'][str(idx)] = {
      'input': inputs,
      'gold_output': outputs
    }
  json.dump(codet5_input, open(output_file, 'w'), indent=2)

def codet5_finetune_output(input_file, output_file, model_dir, model_name, index, num_output=10):
  tokenizer = RobertaTokenizer.from_pretrained('/'.join(model_dir.split('/')[:-2]) + '/' + model_name[:-9])
  model = T5ForConditionalGeneration.from_pretrained(model_dir + model_name + '/' + str(index)).to(device_ids[0])

  codet5_output = json.load(open(input_file, 'r'))
  codet5_output['model'] = model_name
  for i in codet5_output['data']:
    text = codet5_output['data'][i]['input']

    print(int(i) + 1, 'generating...')

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
    codet5_output['data'][i]['output'] = output
  json.dump(codet5_output, open(output_file, 'w'), indent=2)

if __name__ == '__main__':
  model_dir = '../../models/Python-finetuned-model/'
  data_dir = '../../data/codeNetPython.jsonl'
  
  model_names = ['codet5-small-finetune', 'codet5-base-finetune', 'codet5-large-finetune']
  os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
  os.environ['CUDA_VISIBLE_DEVICES']="0"
  device_ids = [0]

  import torch
  from transformers import RobertaTokenizer, T5ForConditionalGeneration

  input_dir = './result/'
  input_file = input_dir + 'codet5_input.json'
  if not os.path.exists(input_dir):
    os.makedirs(input_dir)
  if not os.path.exists(input_file):
    print("==========Preparing input of benchmark to finetuned codet5 model==========")
    codet5_finetune_input(input_file, data_dir=data_dir)
    print("==========Input written to " + input_file)

  for model_name in model_names:
    for i in range(5):
      output_file = './result/' + model_name + '_output' + str(i) + '.json'
      print("==========Generating output of benchmark by " + model_name + "==========")
      codet5_finetune_output(input_file, output_file, model_dir, model_name, i, num_output=10)
      print("==========Output written to " + output_file)


