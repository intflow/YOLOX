import sys
import os
import re
import cv2
import numpy as np
import math
import yaml
import pandas as pd
import pandasql as ps
import time

script_path = os.path.dirname(__file__)
os.chdir(script_path)

def rotate(origin, point, radian):
    ox, oy = origin 
    px, py = point
    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)
    return round(qx), round(qy)

def rotate_box_dot(x_cen, y_cen, width, height, theta):

    x_min = x_cen-width/2
    y_min = y_cen-height/2
    rotated_x1,rotated_y1=rotate((x_cen,y_cen),(x_min,y_min),theta)
    rotated_x2,rotated_y2=rotate((x_cen,y_cen),(x_min,y_min+height),theta)
    rotated_x3,rotated_y3=rotate((x_cen,y_cen),(x_min+width,y_min+height),theta)
    rotated_x4,rotated_y4=rotate((x_cen,y_cen),(x_min+width,y_min),theta)

    answer_dict_ = {
        "Rx" : np.array([rotated_x1, rotated_x2, rotated_x3, rotated_x4]),
        "Ry" : np.array([rotated_y1, rotated_y2, rotated_y3, rotated_y4])
    }

    return answer_dict_

def roi_in_box(img_width, img_height, rbox_dict, width_pad, height_pad):
    min_x = rbox_dict["xmin"]
    min_y = rbox_dict["ymin"]
    max_x = rbox_dict["xmax"]
    max_y = rbox_dict["ymax"]

    width_boundary = (img_width * width_pad, img_width * (1-width_pad))
    height_boundary = (img_height * height_pad, img_height * (1-height_pad))

    if min_x >= width_boundary[0] and max_x <= width_boundary[1] and \
        min_y >= height_boundary[0] and max_y <= height_boundary[1]:
        return True
    else:
        return False

def cutter_fix_45(xc, yc, width, height):
    f_dist = (width + height) / np.sqrt(2)
    answer_dict_ = {
        "xmin" : int(xc - f_dist/2),
        "ymin" : int(yc - f_dist/2),
        "xmax" : int(xc + f_dist/2),
        "ymax" : int(yc + f_dist/2)
    }
    return answer_dict_

def cutting_test(config):

    # file_name_list = [x for x in sorted(os.listdir(config['det_folder_path'])) if re.search(config['det_format'], x) is not None]
    # det = pd.read_csv(os.path.join(config['det_folder_path'], file_name_list[0]), sep=",",encoding="utf-8")

    #? 2.5 second
    if config["column"]:
        det = pd.read_csv(config['trk_file_path'], sep=",",encoding="utf-8")
    else:
        det = pd.read_csv(config['trk_file_path'], sep=",",encoding="utf-8", \
            names=["frame","x","y","w","h","t","no_x","no_y","ne_x","ne_y","ta_x","ta_y","trk"], header=None)
    frame_list = ps.sqldf("select distinct frame from det")
    frame_list = np.fromiter(frame_list.to_dict()['frame'].values(), dtype=int)

    cap = cv2.VideoCapture(config['video_file_path'])

    if not os.path.isdir(os.path.join(config['output_path'])):
        os.mkdir(os.path.join(config['output_path']))

    if not os.path.isdir(os.path.join(os.path.dirname(config['video_file_path']), "trk_img")):
        os.mkdir(os.path.join(os.path.dirname(config['video_file_path']), "trk_img"))

    for frm in frame_list:

        start = time.time()

        #! step1 : 0.01초 ------------------------------------------------------------------------------
        if cap.isOpened() == False:
            break

        # frame_det = ps.sqldf("select * from det where frame = {}".format(frm))
        frame_det = det[det['frame'] == frm]
        frame_in_trk_list = frame_det['trk'].values
        
        ret, img = cap.read()
        img_h, img_w, _ = img.shape

        #? mask code
        mask_total = np.zeros((img.shape[0], img.shape[1]))

        #! step1 : 0.01초 ------------------------------------------------------------------------------
        # print("frame info : {}".format(time.time() - start))

        for trk in frame_in_trk_list:

            step2 = time.time()
            #! step2 : 0.005초 ------------------------------------------------------------------------------
            #? init
            mask_ = np.zeros((img.shape[0], img.shape[1]))
        
            #! landmarks는 사용하지 않으니, json에서 중복된 사진도 사용하자~
            # frame_trk_det = ps.sqldf("select x,y,w,h,t,frame,trk from frame_det where frame = {} and trk = {}".format(frm, trk))
            # frame_trk_det = ps.sqldf("select distinct x,y,w,h,t,frame,trk from frame_det where frame = {} and trk = {}".format(frm, trk))
            frame_trk_det = frame_det[frame_det['trk'] == trk]
            
            if len(frame_trk_det['trk']) != 1:
                print("ERROR : frm : {} / trk : {}".format(frm, trk))
                continue

            x_cen  = int(frame_trk_det['x'])
            y_cen  = int(frame_trk_det['y'])
            width  = int(frame_trk_det['w'])
            height = int(frame_trk_det['h'])
            theta  = float(frame_trk_det['t'])
            # Rdot = rotate_box_dot(x_cen, y_cen, width, height, theta)
            
            rdot = rotate_box_dot(x_cen,y_cen,width,height,theta)
            Rfdot = cutter_fix_45(x_cen, y_cen, width, height)

            # print("trk info : {}".format(time.time() - step2))
            #! step2  ------------------------------------------------------------------------------

            if roi_in_box(img_w, img_h, Rfdot, config['width_roi_padding'], config['height_roi_padding']):
                ## rotated_box
                

                polygon1 = np.array([[rdot['Rx'][0], rdot['Ry'][0]], [rdot['Rx'][1], rdot['Ry'][1]], \
                    [rdot['Rx'][2], rdot['Ry'][2]], [rdot['Rx'][3], rdot['Ry'][3]]], np.int32)
                cv2.fillPoly(mask_total, [polygon1],(255))
                cv2.fillPoly(mask_, [polygon1],(255))

                ## 이걸 자르면 됨
                # img = cv2.rectangle(img, (Rfdot['xmin'], Rfdot['ymin']), (Rfdot['xmax'], Rfdot['ymax']), (255,0,0), 2 )
                # cv2.imwrite( "test1.jpg", img)
                img_cut = img[Rfdot['ymin']:Rfdot['ymax'], Rfdot['xmin']:Rfdot['xmax'],:].copy()
                mask_cut = mask_[Rfdot['ymin']:Rfdot['ymax'], Rfdot['xmin']:Rfdot['xmax']].copy()
                img_cut[mask_cut != 255] = 0
                
                pig_img = cv2.resize(img_cut, (config['width'],config['height']), interpolation=cv2.INTER_LINEAR)
                # pig_img_folder = "{}_{}".format(os.path.splitext(os.path.basename(config['video_file_path']))[0], str(trk).zfill(3))
                pig_img_folder = "{}".format(str(trk).zfill(3))
                pig_img_filename = "{}.jpg".format(str(frm).zfill(5))
                
                
                if not os.path.isdir(os.path.join(os.path.dirname(config['video_file_path']), "trk_img", pig_img_folder)):
                    os.mkdir(os.path.join(os.path.dirname(config['video_file_path']), "trk_img", pig_img_folder))
                
                #? NOTE 0.5초 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                cv2.imwrite( os.path.join(os.path.dirname(config['video_file_path']), "trk_img", pig_img_folder, pig_img_filename), pig_img)
                #? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                

        print("{} | {} [time : {}]".format(frm, len(frame_list), (time.time() - start)))  

        if not os.path.isdir(os.path.join(os.path.dirname(config['video_file_path']), "total_img")):
            os.mkdir( os.path.join(os.path.dirname(config['video_file_path']), "total_img") )

        # if not os.path.isdir(os.path.join(config['output_path'], "total_img")):
        #     os.mkdir(os.path.join(config['output_path'], "total_img"))

        img[mask_total != 255] = 0
        cv2.imwrite(os.path.join(os.path.dirname(config['video_file_path']), "total_img", "{}.jpg".format(str(frm).zfill(5))), img)


    # img = cv2.rectangle(img, (int(img_w*config['width_roi_padding']), int(img_h*config['height_roi_padding'])), \
    #     (int(img_w*(1-config['width_roi_padding'])), int(img_h*(1-config['height_roi_padding']))), (255,0,255), 4 )
    # img_resize = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("test", img_resize)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    with open('cutter_f1_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    cutting_test(config)
