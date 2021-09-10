import sys
import os
import re
import cv2
import numpy as np
import math
import yaml
import pandas as pd
import pandasql as ps
import shutil


def main(config):
    
    for folder_name in sorted(os.listdir(config['cutting_img']['output_path'])):
        folder_path = os.path.join(config['cutting_img']['output_path'], folder_name)
        
        memory_list = []
        
        for filename in sorted(os.listdir(folder_path)):
            if len(memory_list) < 5:
                memory_list.append(filename)
            else:
                start = re.split("[.]", memory_list[0])[0]
                end = re.split("[.]", memory_list[3])[0]
                exp = re.split("[.]", memory_list[4])[0]
                
                # 연속된 조건문
                if int(start) + 4 == int(exp):
                    # 연속된 사진 인식
                    re_foldername = re.split("/", folder_path)[-1] + "_" + start + "_" + end + "_" + exp
                    result_forder_name = os.path.join(config["output_refile_path"],re_foldername)
                    
                    if not os.path.isdir(result_forder_name):
                        os.mkdir(result_forder_name)
                        for fil in memory_list:
                            src = os.path.join(folder_path, fil)
                            dsc = os.path.join(result_forder_name, fil)
                            shutil.copy(src, dsc)
                        memory_list.pop(0)
                        memory_list.append(filename)
                    else:
                        for fil in memory_list:
                            src = os.path.join(folder_path, fil)
                            dsc = os.path.join(result_forder_name, fil)
                            shutil.copy(src, dsc)
                        memory_list.pop(0)
                        memory_list.append(filename)         
                else:
                    # 연속적이지 않은 사진
                    pass
        
        print("forder name : {}".format(folder_name))       
        # break


def zfill_name(config):
    for folder_name in sorted(os.listdir(config['cutting_img']['output_path'])):
        folder_path = os.path.join(config['cutting_img']['output_path'], folder_name)
        
        for name in os.listdir(folder_path):
            zfill_name = re.split("[.]", name)[0].zfill(4) + "." + re.split("[.]", name)[1]
            print("zfill : {} | origin : {}".format(zfill_name, name))
            os.rename(os.path.join(folder_path, name), os.path.join(folder_path, zfill_name))


if __name__ == '__main__':
    with open('cutter_f1_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # file name 전처리
    if config['re_filename']:
        zfill_name(config)
    
    # main process
    main(config)

