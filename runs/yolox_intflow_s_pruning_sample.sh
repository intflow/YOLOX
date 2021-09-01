#!/bin/bash

#python3 tools/pruning.py -expn yolox_s_oad_lm3__intflow_total_1K -n yolox_s_oad_lm3 -f exps/yolox_oad_lm3__intflow_total_1K/yolox_s_oad_lm3.py -p p0 \
#-c /data/pretrained/hcow/yolox_s_oad_lm3__intflow_total_100K_2_test1.pth

python3 tools/train.py -expn yolox_s_oad_lm3__intflow_total_1K -n yolox_s_oad_lm3 -f exps/yolox_oad_lm3__intflow_total_1K/yolox_s_oad_lm3.py -d 4 -b 32 --cache --fp16 --resume \
-c YOLOX_outputs/yolox_s_oad_lm3__intflow_total_1K/p0_ckpt.pth

##python3 tools/train.py -expn yolox_s_oad_lm3__intflow_total_1K -n yolox_s_oad_lm3 -f exps/yolox_oad_lm3__intflow_total_1K/yolox_s_oad_lm3.py -d 4 -b 32 --cache --fp16 \
##-c /data/pretrained/yolox_s.pth
##
##python3 tools/train.py -expn yolox_s_oad_lm3__intflow_total_1K -n yolox_s_oad_lm3 -f exps/yolox_oad_lm3__intflow_total_1K/yolox_s_oad_lm3.py -d 4 -b 32 --cache --fp16 \
##-c /data/pretrained/yolox_s.pth