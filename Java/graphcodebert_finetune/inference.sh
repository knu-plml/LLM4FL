lang=java #programming language
beam_size=10
batch_size=10
source_length=512
target_length=128
output_dir=./defects4j_result/
test_file=../../data/defects4j_test_input.jsonl
pretrained_model=../../models/graphcodebert-base
test_model=../../models/Java-finetuned-model/graphcodebert-base-finetune/

mkdir -p $output_dir

python run.py --do_test --iterN $iterN --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name $pretrained_model --config_name $pretrained_model --load_model_path $test_model --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=./humaneval_result/
test_file=../../data/humaneval_test_input.jsonl

mkdir -p $output_dir

python run.py --do_test --iterN $iterN --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name $pretrained_model --config_name $pretrained_model --load_model_path $test_model --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

output_dir=./quixbugs_result/
test_file=../../data/quixbugs_test_input.jsonl

mkdir -p $output_dir

python run.py --do_test --iterN $iterN --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name $pretrained_model --config_name $pretrained_model --load_model_path $test_model --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
