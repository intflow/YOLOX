#!/bin/bash

/opt/conda/bin/python ../tools/train.py -expn crowd -n crowdhuman -f ../exps/intflow_total_1K/crowdhuman.py -d 1 -b 64 --fp16 -o -c /data/pretrained/hcow/yolox_intflow_s-intflow_total_100K.pth.tar

