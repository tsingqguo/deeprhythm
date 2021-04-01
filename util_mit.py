import numpy as np
import dlib
import cv2
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calculate_ROI(img, block_h, block_w):
    height, width, channels = np.shape(img)
    blk_h, blk_w = int(height/block_h), int(width/block_w)
    roi_seg = np.zeros((block_h*block_w, channels))
    for bh in range(block_h):
        for bw in range(block_w):
            roi_seg[bh*block_w+bw] = [np.average(img[blk_h*bh:blk_h*(bh+1),blk_w*bw:blk_w*(bw+1),i]) for i in range(3)]
            # roi_seg[bh][bw] = [np.average(img[blk_h*bh:blk_h*(bh+1),blk_w*bw:blk_w*(bw+1),i]) for i in range(3)]
    return roi_seg

def reshape_ROI_SEG(roi_seg):
    bh, bw, channels = np.shape(roi_seg)
    frame_seg = np.zeros((bh*bw, channels))
    for h in range(bh):
        for w in range(bw):
            frame_seg[h*bw+w] = roi_seg[h][w]
    # print(frame_seg)
    # print(SSSSSSSSSSSSSSSS)
    return frame_seg

def get_frame_seg(img, ROI_h, ROI_w):

    ROI_seg = calculate_ROI(img, ROI_h, ROI_w)

    # frame_seg = reshape_ROI_SEG(ROI_seg)

    return ROI_seg

def normalization(st_map):
    # print([x[5, 1] for x in st_map])
    time, block, channels = np.shape(st_map)
    nom_map = np.zeros((time, block, channels))
    for blk in range(block):
        for c in range(channels):
            t_seq = [x[blk, c] for x in st_map]
            for t in range(time):
                nom_map[t][blk][c] = (t_seq[t] - min(t_seq)) / (max(t_seq) - min(t_seq)) * 255
    # print([x[5, 1] for x in nom_map])
    return nom_map
