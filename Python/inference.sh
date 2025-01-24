#!/bin/bash

python inference.py 512 Python graphcodebert-base 0 0 
python inference.py 1024 Python unixcoder-base 0 0 

python inference.py 512 Python codet5-small 1 0 &

python inference.py 1024 Python codegen-350M 1 0 
#python inference.py 1024 Python incoder-2B 0 0
