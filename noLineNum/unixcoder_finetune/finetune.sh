lang=java #programming language
lr=5e-5
batch_size=10
accm_steps=1
beam_size=10
source_length=768
target_length=128
output_dir=../../models/noLineNum-finetuned-model/unixcoder-base-finetune
train_file=../../data/finetune_training.jsonl
dev_file=empty
epochs=1
pretrained_model=../../models/unixcoder-base/

python run.py \
--do_train \
--do_eval \
--seed 0 \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs

python run.py \
--do_train \
--do_eval \
--seed 1 \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs

python run.py \
--do_train \
--do_eval \
--seed 2 \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs

python run.py \
--do_train \
--do_eval \
--seed 3 \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs

python run.py \
--do_train \
--do_eval \
--seed 4 \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs
