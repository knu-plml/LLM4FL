lr=1e-4
batch_size=20
beam_size=10
source_length=512
target_length=128
output_dir=../../models/Java-finetuned-model/graphcodebert-base-finetune
train_file=../../data/finetune_training.jsonl
dev_file=empty
epochs=1
pretrained_model=../../models/graphcodebert-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name $pretrained_model --config_name $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs

