import numpy as np
import pandas as pd
import glob
import cv2, batch_generate
from math import *
def coord_judge(c, m):
    if c < 0:
        return 0
    elif(c > m):
        return m
    else:
        return c
default_w = 960
default_h = 640
file_name = glob.glob('./FDDB-folds/*-ellipseList.txt')
#print(file_name)
#df = pd.DataFrame(columns = ['Frame', 'ra', 'rb', 'theta', 'cx', 'cy', 's','FileName'])
df = pd.DataFrame(columns = ['Frame', 'xmin', 'ymin', 'xmax', 'ymax', 's','FileName'])
df2 = pd.DataFrame(columns = ['Frame', 'x_center', 'y_center', 'w', 'h', 's','FileName'])
for name in file_name:
    #print(name)
    f = open(name, 'r')
    total_lines = len(f.readlines())
    f = open(name, 'r')
    count = 0
    for line in open(name):
        if(count == total_lines):
            break
        #print(f.readline())
        #according to FDDB description, first line is img name, second line is number of faces,
        name_line = f.readline()
        Frame = name_line.split('\n')
        img_loc = './originalPics/' + Frame[0] + '.jpg'
        img = cv2.imread(img_loc)
        h, w = np.shape(img)[0:2]
        #number of faces
        num_img_faces = f.readline()
        count +=2
        for i in range(int(num_img_faces)):
            annotation = f.readline()
            b = annotation.split(' ')
            #print('annotation data is ', b)
            count += 1
            #the structure of this annotation data is ra, rb, theta, cx, cy and s(confidence score,all 1 here)
            s = 1
            ra = np.round(float(b[0]))
            rb = np.round(float(b[1]))
            theta = np.round(float(b[2]))
            cx = np.round(float(b[3]))
            cy = np.round(float(b[4]))
            print(ra,rb,theta,cx,cy)
            #then convert to rectangle(center x, center y, w, h structure)
            tan_ = -(rb/ra) * tan(theta)
            t = atan(tan_)
            x1 = cx + (ra * cos(t) * cos(theta) - rb * sin(t) * sin(theta))
            x2 = cx + (ra * cos(t + pi) * cos(theta) - rb * sin(t + pi) * sin(theta))
            print(x1,x2)
            xmax = coord_judge(max(x1,x2), w)
            xmin = coord_judge(min(x1,x2), w)

            if(tan(theta) != 0 ):
                tan_ = (ra/rb) * (1/tan(theta))
            else:
                tan_ = (ra/rb) * (1/(tan(theta) + 0.0001))

            t = atan(tan_)
            y1 = cy + (rb * sin(t) * cos(theta) + ra * cos(t) * sin(theta))
            y2 = cy + (rb * sin(t + pi) * cos(theta) + ra * cos(t + pi) * sin(theta))
            ymax = coord_judge(max(y1,y2), h)
            ymin = coord_judge(min(y1,y2), h)
            data2 = {'Frame':Frame[0]+'.jpg' ,'xmin':xmin ,'ymin':ymin ,'xmax':xmax ,'ymax':ymax ,'s':s ,'FileName':img_loc}
            df = df.append(data2, ignore_index=True)
            print('successful add {} into dataframe'.format(data2))
            #convert xmin, ymin, xmax, ymax to center_x, center_y, w, h
            gta = [xmin, ymin, xmax, ymax]
            gta = np.expand_dims(gta, axis = 0)
            bbox = batch_generate.bbox2cxcy(gta)
            center_x = np.round(float(bbox[:,0])) / w * default_w
            center_y = np.round(float(bbox[:,1])) / h * default_h
            ww = np.round(float(bbox[:,2])) / w * default_w
            hh = np.round(float(bbox[:,3])) / h * default_h
            data3 = {'Frame':Frame[0] + '.jpg','x_center': np.round(center_x), 'y_center': np.round(center_y), 'w': np.round(ww), 'h': np.round(hh), 's':s, 'FileName':img_loc }
            df2 = df2.append(data3, ignore_index=True)

print(df.head())
print(df2.head())
df.to_csv('FDDB.csv')
df2.to_csv('FDDB2XYWH.csv')
