accm_steps=1
lang=Python
beam_size=10
epochs=1
batch_size=3
source_length=768
target_length=128
output_dir=./result/
test_file=../../data/codeNetPython.jsonl
pretrained_model=../../models/unixcoder-base
test_model=../../models/Python-finetuned-model/unixcoder-base-finetune/

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
