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
        if 'gold_output' not in model_output['data'][proj]:
            continue

        if model_output['data'][proj]['gold_output'] == "":
            continue

        if 'output' not in model_output['data'][proj]:
            continue

        #print('start validating', proj, "in", input_file)
        total += 1

        for key, value in model_output['data'][proj].items():
        gold_output = model_output['data'][proj]['gold_output']

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
                    start_index = candidate.rindex(last_input_code) + len(last_input_code)
                except:
                    continue
                if '<|endoftext|>' in candidate:
                    end_index = candidate.index('<|endoftext|>')
            elif 'incoder' in input_file:
                try:
                    start_index = candidate.rindex(last_input_code) + len(last_input_code)
                except:
                    continue
                if '<|endofmask|>' in candidate:
                    end_index = candidate.index('<|endofmask|>')
            else:
                candidate = candidate.strip()
            line = candidate[start_index: end_index].strip()
            if line == '':
                continue
        
            if '-' in gold_output:
                gold_output = gold_output.split('-')[0]

            if '\t' in gold_output:
                gold_output = gold_output.split('\t')[0]
            if '\t' in line:
                line = line.split('\t')[0]
              

            if line == gold_output.strip():
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
    benchmark = ['defects4j', 'humaneval', 'quixbugs']
    for data in benchmark:
        dataPrefixes = list(map(lambda x: './' + x + '_finetune/' + data + '_result/' + x, models))

        topsLists = []
        print(data)
        for dataPrefix in dataPrefixes:
            print(dataPrefix)
            if 'codet5' in dataPrefix:
                dataPrefix += '-small-finetune'
            elif 'codegen' in dataPrefix:
                dataPrefix += '-350M-finetune'
            elif 'incoder' in dataPrefix:
                dataPrefix += '-1B-finetune'
            topLists = []
            for i in range(5):
                input_file = dataPrefix + '_output' + str(i) + '.json'
                output_file = dataPrefix + '_validate' + str(i) + '.json'
                topList = validate_python(input_file)

                #print(topList)
                topLists.append(topList)

            topLists = np.sort(np.array(topLists), axis=0)[2:]
            print(np.mean(topLists, axis=0))
        print()
