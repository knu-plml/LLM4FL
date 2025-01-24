#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python finetune.py incoder-1B 0
CUDA_VISIBLE_DEVICES=2,3 python finetune.py incoder-1B 1
CUDA_VISIBLE_DEVICES=2,3 python finetune.py incoder-1B 2
CUDA_VISIBLE_DEVICES=2,3 python finetune.py incoder-1B 3
CUDA_VISIBLE_DEVICES=2,3 python finetune.py incoder-1B 4
