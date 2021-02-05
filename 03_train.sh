#!/bin/bash

PRETRAIN=./models/mbart.cc25/model.pt 	#Path to the pretrained mBART Model
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SRC=src
TGT=label
NAME=$1
DATADIR=./postprocessed/${NAME}	
SAVEDIR=./checkpoint_${NAME}

fairseq-train ${DATADIR} --encoder-normalize-before --decoder-normalize-before \
--arch mbart_large --task translation_from_pretrained_bart --layernorm-embedding \
--source-lang ${SRC} --target-lang ${TGT} \
--criterion label_smoothed_cross_entropy --label-smoothing 0.2 --dataset-impl mmap \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 40000 \
--warmup-updates 2500 --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 768 --update-freq 4 \
--save-interval 1 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 \
--finetune-from-model ${PRETRAIN} --langs ${langs} \
--ddp-backend no_c10d --memory-efficient-fp16 --max-epoch 40 \
--save-dir ${SAVEDIR} \
--tensorboard-logdir ${SAVEDIR}/tensorboard/
