import os 
import sys
import time
import torch
import torch.nn as nn
from datetime import datetime
from dataset import Dataset, custom_collate
from transformers import get_cosine_schedule_with_warmup
from transformers import T5ForConditionalGeneration, RobertaTokenizer, T5Config

def print_time():
    now = datetime.now()
    print('시간:', now)
    print('time:', now.timestamp())


def validation_step(model, validation_loader, save_dir, prev_loss):
    print('start validation')
    print_time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
            loss = output.loss
            validation_loss.append(loss.mean().item())
    currLoss = round(sum(validation_loss) / len(validation_loss), 4)
    print('validation loss:', currLoss)
    if currLoss < prev_loss:
        model.module.save_pretrained(save_dir)
    model.train()
    return currLoss


def fine_tune(training_file, validation_file, epochs, batch_size, save_dir, parallel=False, load_range=None):
    tokenizer = RobertaTokenizer.from_pretrained(vocabulary_file)
    config = T5Config.from_pretrained(pretrained_file)
    if not parallel:
        model = T5ForConditionalGeneration(config)
        model = nn.DataParallel(model, device_ids=[device_ids[0]]).to(device_ids[0])
    else:
        model = T5ForConditionalGeneration(config, device_map="auto")
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    print('save dir:', save_dir)
    print()
    print('start data pre-processing')
    print_time()
    
    # for CodeGen models, max_length = 768, due to memory and speed limit
    # for CodeT5 models, max_length = 512, due to model configuration limit
    training_dataset = Dataset(training_file, tokenizer, max_length=480, shuffle=False, load_range=load_range)
    validation_dataset = Dataset(validation_file, tokenizer, max_length=480, load_range=load_range)
    training_sampler = torch.utils.data.RandomSampler(training_dataset)
    validation_sampler = torch.utils.data.RandomSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        #dataset=validation_dataset, batch_size=2 * batch_size, shuffle=False,
        dataset=validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )
    print('finish data pre-processing')
    print_time()
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(epochs * len(training_loader))
    )
    evalLoss = 987654321
    earlyStopCount = 0
    for epoch in range(epochs):
        model.train()
        training_loss = []
        start_time = time.time()
        oom = 0
        for i, data in enumerate(training_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            try:
                optimizer.zero_grad()
                output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                loss = output.loss
                
                loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                optimizer.step()
                scheduler.step()
                training_loss.append(loss.mean().item())
            except Exception as e:
                if 'out of memory' in str(e):
                    oom += 1
                model.zero_grad()
                optimizer.zero_grad()
                del data

                torch.cuda.empty_cache()

            if i % 1000 == 0:
                print('epoch: {}, step: {}/{}, loss: {}, lr: {}, oom: {}, time: {}s'.format(
                    epoch + 1, i, len(training_loader),
                    round(sum(training_loss) / len(training_loss), 4),
                    round(scheduler.get_last_lr()[0], 6), oom,
                    int(time.time() - start_time)
                ))
                start_time = time.time()
                oom = 0

                currEvalLoss = validation_step(model, validation_loader, save_dir, evalLoss)
                if evalLoss > currEvalLoss:
                    evalLoss = currEvalLoss
                    earlyStopCount = 0
                else:
                    earlyStopCount += 1
                print()
                if earlyStopCount == 5:
                    break
        validation_step(model, validation_loader, save_dir, evalLoss)


if __name__ == '__main__':
    model_name = 'codet5-small'
    iterN = '0'

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if len(sys.argv) > 2:
            iterN = sys.argv[2]

    training_file = '../../data/finetune_training.jsonl'          # change to fine-tuning data path
    validation_file = '../../data/finetune_validation.jsonl'      # change to fine-tuning data path
    vocabulary_file = '../../models/' + model_name
    pretrained_file = '../../models/' + model_name

    device_ids = [1]

    if model_name == 'codet5-small':
        batch_size = 40

    fine_tune(
        training_file, validation_file, epochs=1, batch_size=batch_size, save_dir='../../models/noPretrained-finetuned-model/'+model_name+'-finetune/' + iterN + '/', parallel=False, load_range=None
    )
