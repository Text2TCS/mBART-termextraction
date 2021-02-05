#!/bin/bash

CHECKPOINT=./checkpoint/checkpoint_best.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=src
TGT=label
NAME=acl2-termeval
DATADIR=./postprocessed/${NAME}
MODEL=/mnt/d/coding/Text2TCS/fairseq/MBart_termeval_acl2/checkpoint/sentence.bpe.model

fairseq-interactive $DATADIR  --path $CHECKPOINT --source-lang $SRC --target-lang $TGT \
--results-path ./results/ --bpe sentencepiece --sentencepiece-model $MODEL --remove-bpe \
--task translation_from_pretrained_bart --langs $langs --memory-efficient-fp16 --max-tokens 1024
