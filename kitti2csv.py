import numpy as np
import pandas as pd
import glob, cv2

file_address = ['./KITTI/training/label_2/*.txt']

#print(glob.glob(file_address[0]))

files = glob.glob(file_address[0])
default_w = 960
default_h = 640
df = pd.DataFrame(columns = ['Frame', 'x_center', 'y_center', 'w', 'h', 'truncated', 'labels', 'FileName'])

labels_cls = ['Car','Van','Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

def label_mapping(label):
    #idx = 0
    #for cls_ in labels_cls:
    #    if (cls_ == label):
    #        return idx
    #    idx += 1
    for i in range(len(labels_cls)):
        if(labels_cls[i] == label):
            return i

def bbox_resize(xmin, ymin, xmax, ymax, Frame):
    img = cv2.imread(Frame)
    img_h, img_w = np.shape(img)[0:2]
    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2.
    y_center = ymin + h / 2.
    xc = np.round(x_center / img_w * default_w)
    yc = np.round(y_center / img_h * default_h)
    w = np.round(w / img_w * default_w)
    h = np.round(h / img_h * default_h)
    return xc,yc,w,h

#scale all data
for name in files:
    f = open(name, 'r')
    total_lines = len(f.readlines())
    f = open(name, 'r')
    count = 0
    #print(f.readlines())
    File = name.split('.')
    #print(File)
    Frame_ = File[1].split('/')
    Frame = Frame_[-1] + '.png'
    FileName = './KITTI/training/image_2/' + Frame
    for line in open(name):
        if(count == total_lines):
            break
        annotation_lists = f.readline()
        #process for annotation lists
        """
        name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score
        """
        count += 1
        data = annotation_lists.split(' ')
        #print(data)
        xmin = np.round(np.float(data[4]))
        ymin = np.round(np.float(data[5]))
        xmax = np.round(np.float(data[6]))
        ymax = np.round(np.float(data[7]))
        truncated = data[1]
        labels = data[0]
        #convert labels to number type
        labels = label_mapping(labels)
        #print(labels)
        #convert xmin, xmax, ymin. ymax size to default_w, default_h(here is 960, 640)
        #first, find the original image location
        Img_Loc = FileName
        center_x, center_y, w, h = bbox_resize(xmin, ymin, xmax, ymax, Img_Loc)
        data2 = {'Frame': Frame,'x_center':center_x ,'y_center':center_y ,'w':w ,'h':h ,'truncated':truncated ,'labels':labels ,'FileName':FileName}
        print(data2)
        df = df.append(data2, ignore_index=True)
print(df.head())
df.to_csv('KITTI.csv')
