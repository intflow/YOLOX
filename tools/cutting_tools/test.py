f = open("test_{}.txt".format(1), mode="a")
f.write("frame,x,y,w,h,t,noX,noY,neX,neY,taX,taY,trk\n")
# f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(i, int(o_cx), int(o_cy), int(o_width), int(o_height), round(o_radian,2), \
#                                                     int(p_nose_x), int(p_nose_y), \
#                                                     int(p_nect_x), int(p_nect_y), \
#                                                     int(p_tail_x), int(p_tail_y), \
#                                                     origin_2json['objects'][j]['trackingId']))
f.flush()
