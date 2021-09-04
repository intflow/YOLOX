python3 tools/demo.py --demo image -expn yolox_x_oad_lm3__tr2_cow -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__tr2_cow/yolox_x_oad_lm3.py \
-c /data/pretrained/hcow/yolox_x_oad_lm3__tr2_cow.pth \
--path ./assets/cow.jpg \
--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu