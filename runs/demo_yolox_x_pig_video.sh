python3 tools/demo.py --demo video -expn yolox_x_oad_lm3__tr2_pig -n yolox_x_oad_lm3 \
-f exps/yolox_oad_lm3__tr2_pig/yolox_x_oad_lm3.py \
-c /data/pretrained/pig/yolox_x_oad_lm3__tr2_pig_test1.pth \
--path /data/EdgeFarm_pig/video_sample/AIDKR_202109020409.mp4 \
--save_folder /data/yolox_out \
--conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu