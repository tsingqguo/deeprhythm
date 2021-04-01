import os
import json
import dlib
from facenet_pytorch import MTCNN
import numpy as np
import util_img as util
import cv2
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Preparing dlib ... ", end='', flush=True)
detector = dlib.get_frontal_face_detector()
predictor_path = '/EX_STORE/Beauty_app/DeepRhythm/shape_predictor_81_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
print("Done")
print("Preparing MTCNN ... ", end='', flush=True)
mtcnn = MTCNN(thresholds=[0.3, 0.3, 0.3], margin=20, keep_all=True, post_process=False, select_largest=False, device=torch.device('cuda', 0))
print("Done")

# dataset path
dataset_root = '/DATASET/FaceForensics/FaceForensics_dataset/FaceForensics_dataset/'
video_dir_name = "original_sequences/youtube/c23/"
# video_dir_name = "manipulated_sequences/Deepfakes/c23/"
dataset_dir = dataset_root + video_dir_name + 'videos/'
# metadata path
meta_dir = '/EX_STORE/Beauty_app/00_Face_data/' + video_dir_name
if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

'''
    Generate aligned face and remove background
'''
def generate_align_face(data_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vidlist = os.listdir(data_path)
    vidlist.sort()
    for vidname in vidlist:
        vidpath = data_path + vidname

        save_vid_path = meta_dir + vidname[:-4] + '/'
        if os.path.exists(save_vid_path):
            continue
        else:
            os.mkdir(save_vid_path)
        
        try:
            align_face = util.preprocess_video(vidname[:-4], detector, predictor, mtcnn, vidpath, save_vid_path)
        except:
            print("Error caused !!")


if __name__=="__main__":
    generate_align_face(dataset_dir, meta_dir)