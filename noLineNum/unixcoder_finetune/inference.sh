accm_steps=1
lang=java #programming language
beam_size=10
epochs=1
batch_size=3
source_length=768
target_length=128
output_dir=./defects4j_result/
test_file=../../data/defects4j_test_input.jsonl
pretrained_model=../../models/unixcoder-base
test_model=../../models/noLineNum-finetuned-model/unixcoder-base-finetune/

mkdir -p $output_dir

python run.py \
  --do_test \
  --model_name_or_path $pretrained_model \
  --load_model_path $test_model \
  --test_filename $test_file \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --beam_size $beam_size \
  --train_batch_size $batch_size \
  --eval_batch_size $batch_size \
  --gradient_accumulation_steps $accm_steps \
  --num_train_epochs $epochs \

output_dir=./humaneval_result/
test_file=../../data/humaneval_test_input.jsonl

mkdir -p $output_dir

python run.py \
  --do_test \
  --model_name_or_path $pretrained_model \
  --load_model_path $test_model \
  --test_filename $test_file \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --beam_size $beam_size \
  --train_batch_size $batch_size \
  --eval_batch_size $batch_size \
  --gradient_accumulation_steps $accm_steps \
  --num_train_epochs $epochs \

output_dir=./quixbugs_result/
test_file=../../data/quixbugs_test_input.jsonl

mkdir -p $output_dir

python run.py \
  --do_test \
  --model_name_or_path $pretrained_model \
  --load_model_path $test_model \
  --test_filename $test_file \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --beam_size $beam_size \
  --train_batch_size $batch_size \
  --eval_batch_size $batch_size \
  --gradient_accumulation_steps $accm_steps \
  --num_train_epochs $epochs \
