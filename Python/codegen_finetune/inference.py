import os
import sys
import json

def codegen_finetune_input(output_file, data_dir):
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
  codegen_input = {'config': 'finetune', 'data': {}}
  for idx, data in enumerate(new_data):
    inputs, outputs = data['inputs'], data['outputs']
    codegen_input['data'][str(idx)] = {
      'input': inputs,
      'gold_output': outputs
    }
  json.dump(codegen_input, open(output_file, 'w'), indent=2)

def codegen_finetune_output(input_file, output_file, model_dir, model_name, index, num_output=10):
  tokenizer = AutoTokenizer.from_pretrained('/'.join(model_dir.split('/')[:-2]) + '/' + model_name[:-9])
  model = CodeGenForCausalLM.from_pretrained(model_dir + model_name + '/' + str(index), device_map='auto')

  codegen_output = json.load(open(input_file, 'r'))
  codegen_output['model'] = model_name
  for i in codegen_output['data']:
    text = codegen_output['data'][i]['input']

    print(int(i) + 1, 'generating...')

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device_ids[0])
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    if input_ids.size(1) >= 768:
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
      output.append(tokenizer.decode(generated_id, skip_special_tokens=False))
    codegen_output['data'][i]['output'] = output
  json.dump(codegen_output, open(output_file, 'w'), indent=2)

if __name__ == '__main__':
  model_dir = '../../models/Python-finetuned-model/'
  data_dir = '../../data/codeNetPython.jsonl'

  #model_names = ['codegen-6B-finetune', 'codegen-2B-finetune']
  model_names = ['codegen-350M-finetune']
  iterN = int(sys.argv[1])

  input_dir = './result/'
  input_file = input_dir + 'codegen_input.json'
  if not os.path.exists(input_dir):
    os.makedirs(input_dir)
  if not os.path.exists(input_file):
    print("==========Preparing input of benchmark to finetuned codegen model==========")
    codegen_finetune_input(input_file, data_dir=data_dir)
    print("==========Input written to " + input_file)

  for model_name in model_names:
    if model_name == 'codegen-6B-finetune':
      device_ids = [0, 1, 2, 3]
    else:
      os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
      os.environ['CUDA_VISIBLE_DEVICES']="0, 1"
      device_ids = [0, 1]

    import torch
    from transformers import AutoTokenizer, CodeGenForCausalLM

    output_file = './result/' + model_name + '_output' + str(i) + '.json'
    print("==========Generating output of benchmark by " + model_name + "==========")
    codegen_finetune_output(input_file, output_file, model_dir, model_name, iterN, num_output=10)
    print("==========Output written to " + output_file)


