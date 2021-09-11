#!/bin/bash

python3 tools/train.py -expn yolox_oad_lm3__crowdhuman -n yolox_s_oad_lm3 \
-f exps/yolox_oad_lm3__crowdhuman/yolox_s_oad_lm3.py -d 4 -b 32 --fp16 \
-c /data/pretrained/yolox_s.pth
