# Training mBART for termextraction

## Overview

This repository contains the scripts used to finetune mBART for the termextraction task on the ACTER dataset (https://github.com/AylaRT/ACTER).
Use the files to recreate the experiment results.

## Installation

The experiment relies on several external dependecies. Create a python 3 virtual environment and install all dependencies using

```
pip install -r requirements.txt
```
or if you use anaconda
```
conda install --file requirements.txt 
```

This also installs the specific Fairseq version that was used for the experiments, including its requirements.
Alternatively, you can get the latest version of Fairseq here: https://github.com/pytorch/fairseq

As stated by the original Fairseq documentation, for faster training on Nvidia GPUs, install nvidia apex:
``` 
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

Additionally, you should install the CLI Tools of SentencePiece (SPM) for creating the sentence piece tokenized data for Fairseq
Install SPM [here](https://github.com/google/sentencepiece)

For the experiments, **Python 3.6.6** and the **CUDA 10.1 Toolkit** was used. Make sure you acquire the correct pyTorch version for your CUDA Toolkit installation, to avoid version mismatch (for example when building nvidia apex).

Finally, you need to download the mBART Model with its dictionary and SentencePiece model:
```
# download model
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.CC25.tar.gz
```
Put the model files under "./models/mBART.cc25/" or change model location in the scripts.
 
## Usage of the scripts

The scripts present in the root of this repository are numbered from 00 to 04. 
The repository comes with the results of scripts 00 and 01 ("termeval" and "preprocessed").

00. ACTER_dataprep.py Python script: 
Tokenizes and concatenates all files in ACTER per domain and language. Saves results to "termeval" dir in root. Change ACTER variable to point to the ACTER dataset (download here https://github.com/AylaRT/ACTER dataset)
01. train_test_split_termeval_revised Python script:
Creates Train, Val and Test split, saves files to "preprocessed". 
	1. Default split: Train = corp and wind domain. Val = equi. Test = htfl.
	2. Label separator can be defined in script. Default: semicolon (comma)
	3. Language and label separator define dir name: Example en_comma = English data, semicolon separated labels
02. spm.sh Bash script:
Create SentencePiece tokenized and binarized files for processing with fairseq. Saves files to "postprocessed".
CLI argument 1 defines dataset. Example for the English, semicolon separated dataset:
	```
	./spm.sh en_comma
	```
03. train.sh Bash script:
Start training process with Fairseq.
04. gen.sh or interactive.sh Bash scripts:
gen.sh generates terms for all examples in binarized "test" file.
interactive.sh can be used interactively to generate terms from raw text input-sentences.
05. termeval_F1.py python script:
Evaluate F1 score on extracted terms.




