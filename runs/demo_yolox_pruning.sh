python3 tools/demo.py --demo video -expn yolox_s_oad_lm3__intflow_total_1K -n yolox_s_oad_lm3 \
                    -f exps/yolox_oad_lm3__intflow_total_100K_2/yolox_s_oad_lm3.py \
                    -c /data/pretrained/hcow/yolox_s_oad_lm3__intflow_total_1K_p0.pth \
                    -m YOLOX_outputs/yolox_s_oad_lm3__intflow_total_1K/p0_ckpt.pth \
                    --path /data/EdgeFarm_cow/video_sample/moonfarm_densecow_20210720_160508447.mp4 \
                    --pruning --conf 0.6 --nms 0.65 --tsize 640 --save_result --device gpu