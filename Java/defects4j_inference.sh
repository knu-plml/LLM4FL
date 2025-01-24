#!/bin/bash

python defects4j_inference.py 512 Java codebert-base 0 0 &
python defects4j_inference.py 512 Java graphcodebert-base 0 0 &
python defects4j_inference.py 1024 Java unixcoder-base 0 0 &

python defects4j_inference.py 1024 Java plbart-base 1 0 &
python defects4j_inference.py 1024 Java plbart-large 1 0 &

python defects4j_inference.py 512 Java codet5-small 1 0 &
python defects4j_inference.py 512 Java codet5-base 1 0 &
python defects4j_inference.py 512 Java codet5-large 0 0 &

python defects4j_inference.py 1024 Java codegen-350M 1 0 


#python defects4j_inference.py 1024 Java codegen-2B 0 0
#python defects4j_inference.py 1024 Java codegen-2B 0 1
#python defects4j_inference.py 1024 Java codegen-2B 0 2
#python defects4j_inference.py 1024 Java codegen-2B 0 3
#python defects4j_inference.py 1024 Java codegen-2B 0 4

#python defects4j_inference.py 1024 Java codegen-6B 0 0
#python defects4j_inference.py 1024 Java codegen-6B 0 1
#python defects4j_inference.py 1024 Java codegen-6B 0 2
#python defects4j_inference.py 1024 Java codegen-6B 0 3
#python defects4j_inference.py 1024 Java codegen-6B 0 4

#python defects4j_inference.py 1024 Java incoder-1B 0 0
#python defects4j_inference.py 1024 Java incoder-1B 0 1
#python defects4j_inference.py 1024 Java incoder-1B 0 2
#python defects4j_inference.py 1024 Java incoder-1B 0 3
#python defects4j_inference.py 1024 Java incoder-1B 0 4

#python defects4j_inference.py 1024 Java incoder-2B 0 0
#python defects4j_inference.py 1024 Java incoder-2B 0 1
#python defects4j_inference.py 1024 Java incoder-2B 0 2
#python defects4j_inference.py 1024 Java incoder-2B 0 3
#python defects4j_inference.py 1024 Java incoder-2B 0 4
