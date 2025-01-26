# Impact of Large Language Models of Code on Fault Localization.

### Setup Instructions
1. Edit configuration:
- Open 'FL.yml' and change 'prefix' section at last line.

2. Create and activate virtual environment:
- Run the following commands in your terminal:
```console
$ conda env create -f FL.yml
$ conda activate FL
$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$ ./download_LLMCs.sh
$ ./setup.sh
```
3. Fine-Tune LLMCs:
- You can now fine-tune LLMCs using our approach.

### Script Descriptions
- `finetune.sh`: Script to fine-tune LLMCs.<br/>
- `inference.sh`: Script to inference using the test benchmark.
- `validate*.py`: Script to evaluate the inference results.

### Directory Structure
The directories are structured as follows:
#### RQ 1.<br/>
- Java

##### RQ 3.<br/>
- Python 
- C-CPP

##### RQ 4.<br/>
(1) w/o both<br/>
- noLineNoPretrained

(2) w/o pre-training<br/>
- noPretrained

(3) w/o line-numbering<br/>
- noLineNum

(4) w/o sequence<br/>
- noSequence
