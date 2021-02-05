#!/bin/bash

BPEMODEL=./models/mbart.cc25/sentence.bpe.model	# Path to SentencePiece BPE Model
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=src
TGT=label
WORKDIR=$1	# Use "." for PWD
NAME=$2
TEST=$3

DATADIR=./postprocessed/${TEST}
RESULTS=${WORKDIR#${WORKDIR%/*}/}/out/${NAME}_${TEST%_*}
CHECKPOINT=${WORKDIR}/checkpoint_${NAME}/checkpoint_last.pt
if [[ $4 == best ]]; then 
	RESULTS=${WORKDIR#${WORKDIR%/*}/}/out_best/${NAME}_${TEST%_*}
	CHECKPOINT=${WORKDIR}/checkpoint_${NAME}/checkpoint_best.pt
fi

fairseq-generate  $DATADIR  --path $CHECKPOINT --source-lang $SRC --target-lang $TGT \
--results-path ${RESULTS} --bpe sentencepiece --sentencepiece-model $BPEMODEL \
--task translation_from_pretrained_bart --langs $langs --max-tokens 1024
