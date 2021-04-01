import os
import cv2
import util_mit as util
import numpy as np

datadir = "/DATASET/__saika_data/dfdc/03_mag_video/"

newviddir = "/DATASET/__saika_data/dfdc/stmap/"

'''
    Use magnified video to produce mmst-map
'''
def generate_mmst_map(mag_path, map_path):
    if not os.path.exists(map_path):
        os.makedirs(map_path)

    ROI_h, ROI_w = 5, 5

    vidlist = os.listdir(mag_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(typename, vidname), end='', flush=True)

        vidpath = mag_path + vidname
        vid = cv2.VideoCapture(vidpath)
        full_st_map = np.zeros((300, 25, 3))
        idx = 0
        while idx < 300:
            success, frame = vid.read()
            if not success:
                break
            # print("  %3d/300 ... " % idx, end='', flush=True)
            frame_seg = util.get_frame_seg(frame, ROI_h, ROI_w)
            full_st_map[idx] = frame_seg
            idx += 1
            # print("Done")
        
        save_vid_path = map_path + vidname
        np.save(save_vid_path + ".npy", full_st_map)
        print("Done")

if __name__=="__main__":
    generate_mmst_map(datadir, newviddir)