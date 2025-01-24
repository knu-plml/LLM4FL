#!/bin/bash

cd /home/shji/FL_LLMCs/Java/unixcoder_finetune > /home/shji/FL_LLMCs/Java/unixcoder_finetune/finetune.log 2>&1

python finetune.py unixcoder-base 0
python finetune.py unixcoder-base 1
python finetune.py unixcoder-base 2
python finetune.py unixcoder-base 3
python finetune.py unixcoder-base 4
