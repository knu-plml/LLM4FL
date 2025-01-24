#!/bin/bash

python defects4j_inference.py 0
python defects4j_inference.py 1
python defects4j_inference.py 2
python defects4j_inference.py 3
python defects4j_inference.py 4

python humaneval_inference.py 0
python humaneval_inference.py 1
python humaneval_inference.py 2
python humaneval_inference.py 3
python humaneval_inference.py 4

python quixbugs_inference.py 0
python quixbugs_inference.py 1
python quixbugs_inference.py 2
python quixbugs_inference.py 3
python quixbugs_inference.py 4
