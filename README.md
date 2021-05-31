# Training mBART for term extraction

## Overview

This repository contains the scripts used to finetune mBART for the termextraction task on the ACTER dataset (https://github.com/AylaRT/ACTER).
Additionally, the repository contains folders named "epoch_XX", which contain the output on the test set of the models trained to XX epoch.
Epoch_40 and Epoch_80 contain folders out and out_best. out is the output of the model at the maximum epoch; out_best the output of the early-stopped model.
See exact number of steps per model under "checkpoint_steps" or see logged training progress with 
```
tensorboard --logdir ./tensorboard/
```

Use the scripts in the root of the repository to recreate these results.



## Installation

The experiment relies on several external dependecies. It is recommended to recreate the experiments on Linux.

Create a python 3 virtual environment and install all dependencies using

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

The scripts present in the root of this repository are numbered from 00 to 05. 
The repository comes with the results of scripts 00 and 01 ("termeval" and "preprocessed").

00. **ACTER_dataprep.py** Python script: 
Tokenizes and concatenates all files in ACTER per domain and language. Saves results to "termeval" dir in root. Change ACTER variable to point to the ACTER dataset (download here https://github.com/AylaRT/ACTER dataset)
01. **train_test_split_termeval_revised.py** Python script:
Creates Train, Val and Test split, saves files to "preprocessed". 
	1. Default split: Train = corp and wind domain. Val = equi. Test = htfl.
	2. Label separator can be defined in script. Default: semicolon (comma)
	3. Language and label separator define dir name: Example en_comma = English data, semicolon separated labels
02. **spm.sh** Bash script:
Create SentencePiece tokenized and binarized files for processing with fairseq. Saves files to "postprocessed".\
CLI argument 1 defines dataset. Example for the English, semicolon separated dataset:
	```
	./spm.sh en_comma
	```
03. **train.sh** Bash script:
Start training process with Fairseq.\ 
Usage: `./train.sh dataset_name` e.g. `./train.sh en_comma`.
04. **gen.sh** or interactive.sh Bash scripts:
gen.sh generates terms for all examples in binarized "test" file.\
Usage: `./gen.sh workdir model_name test_name` e.g. `./gen.sh . en_comma fr_comma`\
interactive.sh can be used interactively to generate terms from raw text input-sentences.
05. **termeval_F1.py** python script:
Evaluate F1 score on extracted terms.

## Results overview

### F1 Scores on ACTER

Training | Test | Precision | Recall | F1
------------ | ------------- | -------------| -------------| -------------|
EN | EN | 45.7 | 63.5 | 53.2
FR | EN | 50.0 | 59.3 | 54.2
NL | EN |  48.3 | 64.3 | 55.2
ALL | EN | 50.2 | 61.6 | 55.3
| | | 
EN | FR | 48.8 | 61.3 | 54.4
FR | FR | 52.7 | 59.6 | 55.9
NL | FR | 54.3 | 60.9 | 57.4
ALL | FR | 55.0 | 60.4 | 57.6
| | | 
EN | NL | 48.8 | 63.9 | 55.4
FR | NL | 56.2 | 63.4 | 59.6
NL | NL | 60.6 | 70.7 | 65.2
ALL | NL | 60.6 | 70.0 | 64.9

### F1 Scores on ACL RD-TEC 2.0 (60/20/20 split as described in paper)
Data Type | Precion | Recall | F1 | 
------------ | ------------- |  ------------- |  ------------- |
Annotator 1 | 73.2 | 77.2 | 75.2 | 
Annotator 2 | 79.4 | 80.7 | 80.0 |






