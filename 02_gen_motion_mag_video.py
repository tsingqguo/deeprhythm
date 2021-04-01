import os
from python_eulerian_video_magnification.magnifycolor import MagnifyColor
from python_eulerian_video_magnification.metadata import MetaData
from python_eulerian_video_magnification.mode import Mode
import cv2
import numpy as np

'''
    Magnification video
'''
def generate_mag_video(vid_path, mag_path):
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)

    vidlist = os.listdir(vid_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(video_dir_name, vidname), end='', flush=True)
        vidpath = vid_path + vidname

        save_vid_path = mag_path + vidname
            
        MagnifyColor(MetaData(file_name=vidpath, low=0.833, high=2, levels=1,
                    amplification=10, target_path=save_vid_path, output_folder='', mode=Mode.COLOR, suffix='color')).do_magnify()
        print("Done")

if __name__=="__main__":

    # video_dir_name = "original_sequences/youtube/c23/"
    video_dir_name = "manipulated_sequences/Deepfakes/c23/"

    data_root_dir = "/EX_STORE/Beauty_app/01_2_align_video/"
    datadir = data_root_dir + video_dir_name

    new_vid_root_dir = "/EX_STORE/Beauty_app/02_mag_video/"
    newviddir = new_vid_root_dir + video_dir_name
    if not os.path.exists(newviddir):
        os.makedirs(newviddir)
        
    generate_mag_video(datadir, newviddir)