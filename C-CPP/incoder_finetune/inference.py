import os
import sys
import json

def incoder_finetune_input(output_file, data_dir):
  dataList = json.load(open(data_dir))
  new_data = []
  for data in dataList:
    elems = ['code', 'code_after']
    inputs, outputs = '', ''
    number = 1

    for s, t in zip(data[elems[0]].strip().split('\n'), data[elems[1]].strip().split('\n')):
      s = s.strip()
      t = t.strip()
      if s == '':
        continue
      inputs += str(number) + '\t' + s + '\n'
      if s != t:
        outputs += str(number) + '\t' + s + '\n'
      number += 1
    n_data = {}
    n_data['inputs'] = inputs
    n_data['outputs'] = outputs
    new_data.append(n_data)

  for i in range(5):
    if not os.path.exists(output_file + str(i) + '.json'):
      temp_data = new_data[int(len(new_data) * (i * 0.2 + 0.1)):int(len(new_data) * (i + 1) * 0.2)]
      incoder_input = {'config': 'finetune', 'data': {}}
      for idx, data in enumerate(temp_data):
        inputs, outputs = data['inputs'], data['outputs']
        incoder_input['data'][str(idx)] = {
          'input': inputs,
          'gold_output': outputs
        }
      json.dump(incoder_input, open(output_file + str(i) + '.json', 'w'), indent=2)

def incoder_finetune_output(input_file, output_file, idx, model_dir, model_name, num_output=10):
  tokenizer = AutoTokenizer.from_pretrained('/'.join(model_dir.split('/')[:-2]) + '/' + model_name[:-9])
  model = AutoModelForCausalLM.from_pretrained(model_dir + model_name + '/' + str(idx), device_map='auto')

  incoder_output = json.load(open(input_file + str(idx) + '.json', 'r'))
  incoder_output['model'] = model_name
  for i in incoder_output['data']:
    text = incoder_output['data'][i]['input']

    print(int(i) + 1, 'generating...')

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device_ids[0])
    eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')

    if input_ids.size(1) >= 1024:
      print('too long:', input_ids.size(1))
      continue

    try:
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
    incoder_output['data'][i]['output'] = output
  json.dump(incoder_output, open(output_file + str(idx) + '.json', 'w'), indent=2)

if __name__ == '__main__':
  model_dir = '../../models/CandCP-finetuned-model/'
  data_dir = '../../data/CVEfixesCandCP.jsonl'
  model_names = ['incoder-1B-finetune', 'incoder-6B-finetune']

  input_dir = './result/'
  input_file = input_dir + 'incoder_input'
  iterN = int(sys.argv[1])
  if not os.path.exists(input_dir):
    os.makedirs(input_dir)
  print("==========Preparing input of benchmark to finetuned incoder model==========")
  incoder_finetune_input(input_file, data_dir=data_dir)
  print("==========Input written to " + input_file, 0, 1, 2, 3, 4, 'json files')

  for model_name in model_names:
    if model_name == 'incoder-6B-finetune':
      device_ids = [0, 1, 2, 3]
    else:
      os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
      os.environ['CUDA_VISIBLE_DEVICES']="0, 1, 2"
      device_ids = [0, 1, 2]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    output_file = './result/' + model_name + '_output'
    print("==========Generating output of benchmark by " + model_name + "==========")
    incoder_finetune_output(input_file, output_file, iterN, model_dir, model_name, num_output=10)
    print("==========Output written to " + output_file)


