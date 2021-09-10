import json
import argparse
import os
import cv2
from natsort import natsorted
import math
import random
from shapely.geometry import Point, Polygon
import pandas

def file_list(file_path):
    f_list =[]
    for each_file in os.listdir(file_path):
        file_ext = os.path.splitext(each_file)[1]
        if file_ext in ['.jpg', '.png', '.json']:
            f_list.append(os.path.join(file_path, each_file))
    
    f_list = natsorted(f_list)

    return f_list

def inBox(area, coord):
    poly = Polygon(area)
    return coord.intersects(poly)

def rotate(origin, point, radian): 

    ox, oy = origin 
    px, py = point
    
    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)
    
    return qx, qy

def argument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_root_path", type=str, default="/DL_data_big/EdgeFarm_pig/superbAI_track/tracking/")
    ap.add_argument("--json_meta_path",type=str, default="/DL_data_big/EdgeFarm_pig/superbAI_track/tracking/meta/")
    ap.add_argument("--img_folder_path",type=str, default="/DL_data_big/EdgeFarm_pig/SL/piglet/img_per_10frame/")
    # ap.add_argument("--json_root_path", type=str, default=f"Y:/DL_data_big/EdgeFarm_pig/superbAI_track/tracking/")
    # ap.add_argument("--json_meta_path",type=str, default=f"Y:/DL_data_big/EdgeFarm_pig/superbAI_track/tracking/meta/")
    # ap.add_argument("--img_folder_path",type=str, default=f"Y:/DL_data_big/EdgeFarm_pig/SL/piglet/img_per_10frame/")

    args = ap.parse_args()

    return args

def main():
    args = argument()

    json_file_list = file_list(args.json_meta_path)
    
    
    for json_i in range(len(json_file_list)):

        if json_i > 0:
            continue

        img_file_list = file_list(args.img_folder_path+str(json_i+1))

        with open(json_file_list[json_i], "r",encoding="UTF-8") as st_json:
            origin_json = json.load(st_json)

        with open(args.json_root_path+origin_json['label_path'][0], "r",encoding="UTF-8") as st_json2:
            origin_2json = json.load(st_json2)

        f = open("test_det_file_{}.txt".format(json_i+1), mode="a")
        f.write("frame,x,y,w,h,t,noX,noY,neX,neY,taX,taY,trk\n")

        # 이미지 하나
        for i in range(len(img_file_list)):
            img = cv2.imread(img_file_list[i],cv2.IMREAD_UNCHANGED)
            for j in range(len(origin_2json['objects'])):
                for z in range(len(origin_2json['objects'][j]['frames'])):
                    if origin_2json['objects'][j]['frames'][z]['annotation']['coord'].get("points"):
                        continue
                    else:
                        if origin_2json['objects'][j]['frames'][z]['num'] == i:
                            o_cx = origin_2json['objects'][j]['frames'][z]['annotation']['coord']['cx']
                            o_cy = origin_2json['objects'][j]['frames'][z]['annotation']['coord']['cy']
                            o_width = origin_2json['objects'][j]['frames'][z]['annotation']['coord']['width']
                            o_height = origin_2json['objects'][j]['frames'][z]['annotation']['coord']['height']
                            o_radian = math.radians(origin_2json['objects'][j]['frames'][z]['annotation']['coord']['angle'])

                            ox_min = o_cx - (o_width/2)
                            oy_min = o_cy - (o_height/2)

                            rotated_x1,rotated_y1=rotate((o_cx,o_cy),(ox_min,oy_min),o_radian)
                            rotated_x2,rotated_y2=rotate((o_cx,o_cy),(ox_min,oy_min+o_height),o_radian)
                            rotated_x3,rotated_y3=rotate((o_cx,o_cy),(ox_min+o_width,oy_min+o_height),o_radian)
                            rotated_x4,rotated_y4=rotate((o_cx,o_cy),(ox_min+o_width,oy_min),o_radian)

                            COLOR_R = random.randint(0,255)
                            COLOR_G = random.randint(0,255)
                            COLOR_B = random.randint(0,255)

                            for k in range(len(origin_2json['objects'])):
                                for t in range(len(origin_2json['objects'][k]['frames'])):
                                    if origin_2json['objects'][k]['frames'][t]['num'] == i:
                                        if origin_2json['objects'][k]['frames'][t]['annotation']['coord'].get("points") and origin_2json['objects'][k]['frames'][t].get('multiple') == None:
                                            if origin_2json['objects'][k]['frames'][t]['annotation']['coord']['points'][0].get('state'):
                                                nose_ = origin_2json['objects'][k]['frames'][t]['annotation']['coord']['points'][0]
                                                neck_ = origin_2json['objects'][k]['frames'][t]['annotation']['coord']['points'][1]
                                                tail_ = origin_2json['objects'][k]['frames'][t]['annotation']['coord']['points'][2]
                                                if inBox(((rotated_x1,rotated_y1),
                                                    (rotated_x2,rotated_y2),
                                                    (rotated_x3,rotated_y3),
                                                    (rotated_x4,rotated_y4)),
                                                    Point([nose_['x'],nose_['y']])) \
                                                        and inBox(((rotated_x1,rotated_y1),
                                                    (rotated_x2,rotated_y2),
                                                    (rotated_x3,rotated_y3),
                                                    (rotated_x4,rotated_y4)),
                                                    Point([neck_['x'],neck_['y']])) \
                                                        and inBox(((rotated_x1,rotated_y1),
                                                    (rotated_x2,rotated_y2),
                                                    (rotated_x3,rotated_y3),
                                                    (rotated_x4,rotated_y4)),
                                                    Point([tail_['x'],tail_['y']])):

                                                    p_nose_x = int(nose_['x'])
                                                    p_nose_y = int(nose_['y'])
                                                    p_nect_x = int(neck_['x'])
                                                    p_nect_y = int(neck_['y'])
                                                    p_tail_x = int(tail_['x'])
                                                    p_tail_y = int(tail_['y'])
                                                    
                                                    # img = cv2.circle(img,(int(nose_['x']),int(nose_['y'])),2,(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # img = cv2.circle(img,(int(neck_['x']),int(neck_['y'])),2,(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # img = cv2.circle(img,(int(tail_['x']),int(tail_['y'])),2,(COLOR_B,COLOR_G,COLOR_R),2)

                                                    # img = cv2.line(img,(int(nose_['x']),int(nose_['y'])),(int(neck_['x']),int(neck_['y'])),(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # img = cv2.line(img,(int(neck_['x']),int(neck_['y'])),(int(tail_['x']),int(tail_['y'])),(COLOR_B,COLOR_G,COLOR_R),2)

                                                    # img = cv2.line(img,(round(rotated_x1),round(rotated_y1)),(round(rotated_x2),round(rotated_y2)),(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # img = cv2.line(img,(round(rotated_x2),round(rotated_y2)),(round(rotated_x3),round(rotated_y3)),(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # img = cv2.line(img,(round(rotated_x3),round(rotated_y3)),(round(rotated_x4),round(rotated_y4)),(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # img = cv2.line(img,(round(rotated_x4),round(rotated_y4)),(round(rotated_x1),round(rotated_y1)),(COLOR_B,COLOR_G,COLOR_R),2)
                                                    # cv2.imwrite("test.jpg", img)
                                                    ## text file making
                                                    # f = open("test_det_file_{}.txt".format(json_i+1), mode="a")
                                                    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(i, int(o_cx), int(o_cy), int(o_width), int(o_height), round(o_radian,2), \
                                                        int(p_nose_x), int(p_nose_y), \
                                                        int(p_nect_x), int(p_nect_y), \
                                                        int(p_tail_x), int(p_tail_y), \
                                                        origin_2json['objects'][j]['trackingId']))
                                                    f.flush()

            # cv2.imwrite("{}.jpg".format(i), img)
            if i>= 600:
                print("break")
                break
        # if json_i == 1:
        #     print("break2")
        #     break

        f.close()

if __name__ == "__main__":
    main()