import numpy as np
import glob
import pandas as pd
import cv2
from utils import box_iou
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def read_csv(data_loc ,file_name, DEBUG = False):
    df = pd.read_csv(file_name)
    #drop first data
    df = df.drop('Unnamed: 0', axis = 1)
    #add image location
    df['FileName'] = data_loc + df['Frame']
    #read img size and resize to default size
    #img = cv2.imread()
    if(DEBUG):
        print(df.head())
    return df

def bbox_transform(df, DEBUG = False):
    """
    transform xmin, ymin, xmax, ymax yo x_center, y_center, w, h
    """
    xmin = df['xmin']
    ymin = df['ymin']
    xmax = df['xmax']
    ymax = df['ymax']
    #print(xmin)
    #convert to x center, y center, w, h
    df['x_center'] = xmin + (xmax - xmin) / 2
    df['y_center'] = ymin + (ymax - ymin) / 2
    df['w'] = xmax - xmin
    df['h'] = ymax - ymin
    if(DEBUG):
        print(df.head())
    return df

def bbox_resize(df, default_w = 960., default_h = 640., DEBUG = False):
    #get unique name
    unique_df = pd.unique(df['Frame'])
    if(DEBUG):
        print(unique_df)
    length = len(unique_df)
    data_frame = pd.DataFrame(columns=['Frame','x_center','y_center','w','h','label','type','FileName'])
    for i in range(length):
        data = df[df['Frame'] == unique_df[i]].reset_index()
        img_loc = data['FileName'][0]
        img = cv2.imread(img_loc)
        img_h, img_w = img.shape[0:2]
        #and resize the bbox
        #data['x_center'] = np.round(data['x_center'] / w * default_w)
        #data['y_center'] = np.round(data['y_center'] / h * default_h)
        #data['w'] = np.round(data['w'] / w * default_w)
        #data['h'] = np.round(data['h'] / h * default_h)
        for j in range(len(data)):
            xmin = data.iloc[j]['xmin']
            ymin = data.iloc[j]['ymin']
            xmax = data.iloc[j]['xmax']
            ymax = data.iloc[j]['ymax']
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2.
            y_center = ymin + h / 2.

            #xc = x_center
            #yc = y_center
            xc = np.round(x_center / img_w * default_w)
            yc = np.round(y_center / img_h * default_h)
            w = np.round(w / img_w * default_w)
            h = np.round(h / img_h * default_h)
            data2 = {'Frame':data.iloc[j]['Frame'],'x_center':xc,'y_center':yc,'w':w,'h':h,'label':data.iloc[j]['label'], 'type':data.iloc[j]['type'], 'FileName':data.iloc[j]['FileName']}
            data_frame = data_frame.append(data2, ignore_index=True)
        #df[df['Frame'] == unique_df[i]]['x_center'] = df[unique_df[i]]['x_center'] / w * default_w
        if(DEBUG):
            #print(len(data))
            #print(data_frame)
            #print(bb_boxes)
            print(img_loc)
            #print(df[df['Frame'] == unique_df[i]]['x_center'])
            print('the image size is w {0}, h {1}'.format(img_w, img_h))
    #data_frame = data_frame.drop('index', axis = 1)
    #data_frame = data_frame.drop('xmin', axis = 1)
    #data_frame = data_frame.drop('xmax', axis = 1)
    #data_frame = data_frame.drop('ymin', axis = 1)
    #data_frame = data_frame.drop('ymax', axis = 1)
    if(DEBUG):
        print(data_frame.head())
    return data_frame

#code was referenced from leetenki/YOLOv2 reimplement
def anchor_box_selected(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h
    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])
    return new_centroids, groups, loss

#main_function to selected the anchor box scales(w, h), k is input value, means the number of anchor box we have
df = read_csv(data_loc = './VOCdevkit2007/VOC2007/JPEGImages/' , file_name = 'voc.csv', DEBUG = True)
#df = bbox_transform(df, DEBUG = True)
df_ = bbox_resize(df, DEBUG = True)
df_.to_csv('voc_xywh.csv')
K = 9
loss_convergence = 1e-5
boxes = []
print(len(df))
for i in range(len(df)):
    x = df_.iloc[i]['x_center']
    y = df_.iloc[i]['y_center']
    w = df_.iloc[i]['w']
    h = df_.iloc[i]['h']
    boxes.append(Box(0, 0, float(w), float(h)))
centroid_indices = np.random.choice(len(boxes), K)
centroids = []
for centroid_index in centroid_indices:
    centroids.append(boxes[centroid_index])
new_centroids, groups, old_loss = anchor_box_selected(K, boxes, centroids)
while(True):
    new_centroids, groups, loss = anchor_box_selected(K, boxes, new_centroids)
    print("loss = %f" % loss)
    if abs(old_loss - loss) < loss_convergence:
        break
    old_loss = loss
# print result
for centroid in centroids:
    print(centroid.w, centroid.h)
