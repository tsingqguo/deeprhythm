import numpy as np
from classifiers import *
from pipeline import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# ---------------------------- data_Meso ------------------------------
'''
    collect meso data
'''

# Load the model and its pretrained weights
classifier = MesoInception4()
classifier.load('/WORKSPACE/ff++/final_data/model/FaceForensics/faceforensics++_models_subset/face_detection/Meso/c23/Deepfakes/weights.h5')

real_video_dir = "/WORKSPACE/ff++/original/youtube/c23/align_original/"
real_vid_list = os.listdir(real_video_dir)
real_vid_list.sort()
real_vid_dict = {x:real_video_dir + x for x in real_vid_list}

fake_video_dir = "/WORKSPACE/ff++/manipulate/DeepFakes/c23/align_original/"
fake_vid_list = os.listdir(fake_video_dir)
fake_vid_list.sort()
fake_vid_dict = {x:fake_video_dir + x for x in fake_vid_list}

video_dict = real_vid_dict
video_dict.update(fake_vid_dict)

data_name_list = real_vid_list + fake_vid_list


# ---------------------------- data_mit ------------------------------
'''
    collect mmst map data
'''

fake_mit_dir = "/WORKSPACE/ff++/manipulate/DeepFakes/c23/data_mit_rgb/"
fake_mit_list = os.listdir(fake_mit_dir)
fake_mit_list.sort()
fake_mit_dict = {x:fake_mit_dir + x + '/' + x + '.npy' for x in fake_mit_list}

real_mit_dir = "/WORKSPACE/ff++/original/youtube/c23/data_mit_rgb/"
real_mit_list = os.listdir(real_mit_dir)
real_mit_list.sort()
real_mit_dict = {x:real_mit_dir + x + '/' + x + '.npy' for x in real_mit_list}

mit_dict = fake_mit_dict
mit_dict.update(real_mit_dict)


# ---------------------------- save ------------------------------
'''
    save meso data and mmst map data into a form that can be used in training
'''

save_path = "/WORKSPACE/ff++/final_data/data_new/df_ytb_c23/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

data_Meso_set = []
data_mit_set = []
data_y_set = []
data_name_set = []

for vid_name in data_name_list:
    print("video {}:".format(vid_name), end='', flush=True)
    if vid_name+".avi" in mit_dict:

        vid_path = video_dict[vid_name] + '/'
        img_list = os.listdir(vid_path)
        if img_list==[]:
            continue
        img_list.sort()

        data_Meso = np.ones(300) * 0.5
        try:
            for img_name in img_list:
                if int(img_name[:-4]) >= 300:
                    continue
                img_path = vid_path + img_name
                img = cv2.resize(cv2.imread(img_path), (256, 256))
                pred = classifier.predict(np.array([img]))
                data_Meso[int(img_name[:-4])] = pred
        except Exception:
            continue
        data_Meso_set.append(data_Meso)

        data_mit = np.load(mit_dict[vid_name+".avi"])
        data_mit_set.append(data_mit)

        if vid_name.find('_')!=-1:
            data_y_set.append(1)
            print(1)
        else:
            data_y_set.append(0)
            print(0)

        data_name_set.append(vid_name)

    print("  ", np.shape(data_Meso_set))
    print("  ", np.shape(data_mit_set))
    print("  ", np.shape(data_y_set))
    print("  ", np.shape(data_name_set))

print("SAVING .... ")
np.save(save_path+"Meso.npy", data_Meso_set)
np.save(save_path+"mit.npy", data_mit_set)
np.save(save_path+"y.npy", data_y_set)
np.save(save_path+"name.npy", data_y_set)