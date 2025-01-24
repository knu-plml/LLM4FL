#!/bin/bash

python finetune.py plbart-base 0
python finetune.py plbart-base 1
python finetune.py plbart-base 2
python finetune.py plbart-base 3
python finetune.py plbart-base 4

python finetune.py plbart-large 0
python finetune.py plbart-large 1
python finetune.py plbart-large 2
python finetune.py plbart-large 3
python finetune.py plbart-large 4
