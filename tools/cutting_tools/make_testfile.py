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
import random


def main(config):
    
    random.seed(19940417)
    
    sorted_folder_list = sorted(os.listdir(config['make_testfile']['input_folder']))
    sample_file_name = random.sample(sorted_folder_list, config['make_testfile']["count"])
    
    for file_name in sample_file_name:
        shutil.move( os.path.join(config['make_testfile']['input_folder'], file_name), \
                     os.path.join(config['make_testfile']['output_folder'], file_name) )
    
if __name__ == '__main__':
    with open('cutter_f1_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # main process
    main(config)

