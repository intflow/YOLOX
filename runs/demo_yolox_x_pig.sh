python3 tools/demo.py --demo image -expn yolox_x_oad_lm3__tr2_pig -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__tr2_pig/yolox_x_oad_lm3.py \
-c YOLOX_outputs/yolox_x_oad_lm3__tr2_pig/best_ckpt.pth \
--path ./assets/pig2.jpg \
--conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu