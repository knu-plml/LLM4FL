lang=C #programming language
beam_size=10
batch_size=5
source_length=512
target_length=128
output_dir=./result/
test_file=../../data/CVEfixesCandCP.jsonl
pretrained_model=../../models/codebert-base
test_model=../../models/CandCP-finetuned-model/codebert-base-finetune/

mkdir -p $output_dir

python run.py --do_test --model_type roberta --model_name_or_path $pretrained_model --load_model_path $test_model --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
