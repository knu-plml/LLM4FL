#!/bin/bash
mkdir models
cd models

git-lfs clone https://huggingface.co/microsoft/codebert-base codebert-base
git-lfs clone https://huggingface.co/microsoft/graphcodebert-base graphcodebert-base

git-lfs clone https://huggingface.co/uclanlp/plbart-base plbart-base
git-lfs clone https://huggingface.co/uclanlp/plbart-large plbart-large

git-lfs clone https://huggingface.co/Salesforce/codet5-small codet5-small
git-lfs clone https://huggingface.co/Salesforce/codet5-base codet5-base
git-lfs clone https://huggingface.co/Salesforce/codet5-large codet5-large

git-lfs clone https://huggingface.co/microsoft/unixcoder-base unixcoder-base

git-lfs clone https://huggingface.co/Salesforce/codegen-350M-multi codegen-350M
git-lfs clone https://huggingface.co/Salesforce/codegen-2B-multi codegen-2B
git-lfs clone https://huggingface.co/Salesforce/codegen-6B-multi codegen-6B

git-lfs clone https://huggingface.co/facebook/incoder-1B incoder-1B
git-lfs clone https://huggingface.co/facebook/incoder-6B incoder-6B

git-lfs clone https://huggingface.co/microsoft/unixcoder-base
git-lfs clone https://huggingface.co/codesage/codesage-base
git-lfs clone https://huggingface.co/Salesforce/codet5p-220m codet5p-220m

