#!/bin/bash

#python finetune.py codegen-6B 0
#python finetune.py codegen-6B 1
#python finetune.py codegen-6B 2
#python finetune.py codegen-6B 3
CUDA_VISIBLE_DEVICES=1,2 python finetune.py codegen-2B 4
