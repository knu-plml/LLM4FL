#!/bin/bash

python quixbugs_inference.py 512 Java graphcodebert-base 0 0 &

python quixbugs_inference.py 512 Java codet5-small 0 0 &

python quixbugs_inference.py 1024 Java unixcoder-base 0 0 &

python quixbugs_inference.py 1024 Java codegen-350M 0 0 &

python quixbugs_inference.py 1024 Java incoder-1B 0 0 &
