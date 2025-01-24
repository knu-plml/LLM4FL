import os 
import sys
import time
import traceback
from dataset import Dataset, custom_collate


def validation_step(model, validation_loader, save_dir, parallel=False):
    print('start validation')
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
    print('validation loss:', round(sum(validation_loss) / len(validation_loss), 4))
    if not parallel:
        model.module.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)
    model.train()


def fine_tune(training_file, validation_file, epochs, batch_size, save_dir, parallel=False, load_range=None):
    tokenizer = RobertaTokenizer.from_pretrained(vocabulary_file)
    if not parallel:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_file)
        model = nn.DataParallel(model, device_ids=[device_ids[0]]).to(device_ids[0])
    else:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_file, device_map="auto")
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    
    # for CodeGen models, max_length = 768, due to memory and speed limit
    # for CodeT5 models, max_length = 512, due to model configuration limit
    training_dataset = Dataset(training_file, tokenizer, max_length=512, shuffle=False, load_range=load_range)
    validation_dataset = Dataset(validation_file, tokenizer, max_length=512, load_range=None)
    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=2 * batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(epochs * len(training_loader))
    )
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
        if i % 10000 == 0 and i > 0:
            validation_step(model, validation_loader, save_dir, parallel=parallel)
    validation_step(model, validation_loader, save_dir, parallel=parallel)


if __name__ == '__main__':
    model_name = 'codet5-small'
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
    device_ids = [0]
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration, RobertaTokenizer
    from transformers import get_cosine_schedule_with_warmup

    training_file = '../../data/finetune_training.jsonl'          # change to fine-tuning data path
    validation_file = '../../data/finetune_validation.jsonl'      # change to fine-tuning data path

    vocabulary_file = '../../models/' + model_name
    pretrained_file = '../../models/' + model_name

    batch_size = 10

    for i in range(5):
        fine_tune(
            training_file, validation_file, epochs=1, batch_size=batch_size, save_dir='../../models/noSequence-finetuned-model/'+model_name+'-finetune/' + str(i) + '/', parallel=False, load_range=None
        )
