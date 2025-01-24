import json
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def validate_python(input_file):
    tops = [0, 0, 0]
    total = 0

    model_output = json.load(open(input_file, 'r'))
    for proj in model_output['data']:
        total += 1
        if 'gold_output' not in model_output['data'][proj]:
            continue

        if model_output['data'][proj]['gold_output'] == "":
            continue

        if 'output' not in model_output['data'][proj]:
            continue

        #print('start validating', proj, "in", input_file)

        for key, value in model_output['data'][proj].items():
        gold_output = int(model_output['data'][proj]['gold_output'].split('\t')[0])

        try:
            last_input_code = model_output['data'][proj]['input'].split('\n')[-2]
        except:
            last_input_code = None
        for rank, candidate in enumerate(model_output['data'][proj]['output'][:10]):
            start_index = 0
            end_index = len(candidate)
            if 'codet5' in input_file:
                if '<s>' in candidate:
                    start_index = candidate.index('<s>') + 3
                    end_index = candidate.index('</s>')
            elif 'codegen' in input_file:
                try:
                    start_index = candidate.index(last_input_code) + len(last_input_code)
                except:
                    continue
                if '<|endoftext|>' in candidate:
                    end_index = candidate.index('<|endoftext|>')
            elif 'incoder' in input_file:
                try:
                    start_index = candidate.index(last_input_code) + len(last_input_code)
                except:
                    continue
                if '<|endofmask|>' in candidate:
                    end_index = candidate.index('<|endofmask|>')
            else:
                candidate = candidate.strip()
            candidate = candidate[start_index: end_index].strip()

            if '\t' in candidate:
                line = candidate.split('\t')[0]
            else:
                line = candidate.split(' ')[0]
            try:
                line = int(line)
                if line > 100:
                    line = -1
            except:
                line = -1

            if line == gold_output:
                if rank <= 1:
                    tops[0] += 1
                if rank <= 3:
                    tops[1] += 1
                if rank <= 5:
                    tops[2] += 1
                break

    #print('Model : ' + input_file.split('_output')[0].split('/')[-1])
    #print('Total :', total)
    #print('Top-1 :', tops[0])
    #print('Top-3 :', tops[1])
    #print('Top-5 :', tops[2])
    #print()
    return tops


if __name__ == '__main__':
    model_dirs = []
    models = ['graphcodebert', 'codet5', 'unixcoder', 'codegen', 'incoder']
    dataPrefixes = list(map(lambda x: './' + x + '_finetune/result/' + x, models))

    topsLists = []
    for dataPrefix in dataPrefixes:
        topLists = []
        if 'codet5' in dataPrefix:
            dataPrefix += '-small-finetune'
        elif 'codegen' in dataPrefix:
            dataPrefix += '-350M-finetune'
        elif 'incoder' in dataPrefix:
            dataPrefix += '-1B-finetune'
        for i in range(5):
            input_file = dataPrefix + '_output' + str(i) + '.json'
            output_file = dataPrefix + '_validate' + str(i) + '.json'

            topList = validate_python(input_file)
            #print(topList)
            topLists.append(topList)

        topLists = np.sort(np.array(topLists), axis=0)[2:]
        print(dataPrefix)
        print(np.mean(topLists, axis=0))
