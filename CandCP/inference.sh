#!/bin/bash

python inference.py 512 CandCP codebert-base 0 0 
python inference.py 1024 CandCP unixcoder-base 0 0 

python inference.py 512 CandCP codet5-small 1 0 &
python inference.py 1024 CandCP codegen-350M 1 0 
#python inference.py 1024 CandCP incoder-1B 0 0
