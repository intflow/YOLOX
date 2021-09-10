import sys
import os
import re
import cv2
import numpy as np
import math
import yaml
import pandas as pd
import pandasql as ps

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

    # file_name_list = [x for x in sorted(os.listdir(config['sample_img']['det_folder_path'])) if re.search(config['sample_img']['det_format'], x) is not None]
    # det = pd.read_csv(os.path.join(config['sample_img']['det_folder_path'], file_name_list[0]), sep=",",encoding="utf-8")

    det = pd.read_csv(config['sample_img']['det_file_path'], sep=",",encoding="utf-8")
    frame_list = ps.sqldf("select distinct frame from det")
    frame_list = np.fromiter(frame_list.to_dict()['frame'].values(), dtype=int)

    for frm in frame_list:

        frame_det = ps.sqldf("select * from det where frame = {}".format(frm))
        frame_dict = frame_det.to_dict()
        frame_in_trk_list = np.fromiter(frame_dict['trk'].values(), dtype=int)
        
        img = cv2.imread(os.path.join(config['sample_img']['img_folder_path'], "{}.jpg".format(frm)))
        img_h, img_w, _ = img.shape

        for trk in frame_in_trk_list:
        
            #! landmarks는 사용하지 않으니, json에서 중복된 사진도 사용하자~
            # frame_trk_det = ps.sqldf("select * from det where frame = {} and trk = {}".format(frm, trk))
            frame_trk_det = ps.sqldf("select distinct x,y,w,h,t,frame,trk from det where frame = {} and trk = {}".format(frm, trk))
        
            if len(frame_trk_det['trk']) != 1:
                print("ERROR : frm : {} / trk : {}".format(frm, trk))
                continue

            x_cen  = int(frame_trk_det['x'])
            y_cen  = int(frame_trk_det['y'])
            width  = int(frame_trk_det['w'])
            height = int(frame_trk_det['h'])
            # theta  = float(frame_trk_det['t'])
            # Rdot = rotate_box_dot(x_cen, y_cen, width, height, theta)
            
            Rfdot = cutter_fix_45(x_cen, y_cen, width, height)
            

            if roi_in_box(img_w, img_h, Rfdot, config['sample_img']['width_roi_padding'], config['sample_img']['height_roi_padding']):
                ## rotated_box
                # img = cv2.line(img,(Rdot['Rx'][0],Rdot['Ry'][0]),(Rdot['Rx'][1],Rdot['Ry'][1]),(0,0,255),2)
                # img = cv2.line(img,(Rdot['Rx'][1],Rdot['Ry'][1]),(Rdot['Rx'][2],Rdot['Ry'][2]),(0,0,255),2)
                # img = cv2.line(img,(Rdot['Rx'][2],Rdot['Ry'][2]),(Rdot['Rx'][3],Rdot['Ry'][3]),(0,0,255),2)
                # img = cv2.line(img,(Rdot['Rx'][3],Rdot['Ry'][3]),(Rdot['Rx'][0],Rdot['Ry'][0]),(0,0,255),2)

                ## 이걸 자르면 됨
                # img = cv2.rectangle(img, (Rfdot['xmin'], Rfdot['ymin']), (Rfdot['xmax'], Rfdot['ymax']), (255,0,0), 2 )
                # cv2.imwrite( "test1.jpg", img)
                pig_img = img[Rfdot['ymin']:Rfdot['ymax'], Rfdot['xmin']:Rfdot['xmax'],:]
                pig_img = cv2.resize(pig_img, (config['cutting_img']['width'],config['cutting_img']['height']), interpolation=cv2.INTER_LINEAR)
                pig_img_folder = "{}_{}".format(re.split("/", config['sample_img']['img_folder_path'])[-1], str(trk).zfill(3))
                pig_img_filename = "{}.jpg".format(str(frm).zfill(4))
                if not os.path.isdir(os.path.join(config['cutting_img']['output_path'], pig_img_folder)):
                    os.mkdir(os.path.join(config['cutting_img']['output_path'], pig_img_folder))
                    cv2.imwrite( os.path.join(config['cutting_img']['output_path'], pig_img_folder, pig_img_filename), pig_img)
                else:
                    cv2.imwrite( os.path.join(config['cutting_img']['output_path'], pig_img_folder, pig_img_filename), pig_img)
                    
    # img = cv2.rectangle(img, (int(img_w*config['sample_img']['width_roi_padding']), int(img_h*config['sample_img']['height_roi_padding'])), \
    #     (int(img_w*(1-config['sample_img']['width_roi_padding'])), int(img_h*(1-config['sample_img']['height_roi_padding']))), (255,0,255), 4 )
    # img_resize = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("test", img_resize)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    with open('cutter_f1_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    cutting_test(config)
