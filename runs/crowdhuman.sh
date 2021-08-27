#!/bin/bash
# hcow/yolox_intflow_s-intflow_total_100K.pth.tar
python3 tools/train.py -expn crowd -n crowdhuman -f exps/crowdhuman/crowdhuman.py -d 1 -b 4 --fp16 -o -c /data/pretrained/yolox_s.pth

