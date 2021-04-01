import dlib
import cv2
import numpy as np
import time
import os

def get_landmark(img, detector, predictor):
    height, width = np.shape(img)[:2]
    # print(height, width)
    # predictor_path = '/WORKSPACE/rPPGNet/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat'

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(predictor_path)

    dets = detector(img, 0)
    landmarks = None
    for k, det in enumerate(dets):
        shape = predictor(img, det)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        for i in range(81):
            for j in range(2):
                if landmarks[i, j]<0:
                    landmarks[i, j] = 0
            if landmarks[i, 1]>height:
                landmarks[i, 1] = height
            if landmarks[i, 0]>width:
                landmarks[i, 0] = width
    return landmarks

def trans_landmark(landmark, trans_mat, height, width):
    trans_land = np.zeros(np.shape(landmark))
    for i in range(np.shape(landmark)[0]):
        trans_land[i] = [landmark[i,0]+trans_mat[0,2], landmark[i,1]+trans_mat[1,2]]
    trans_land = np.array(trans_land, 'int32')
    for i in range(trans_land.shape[0]):
        if trans_land[i, 0]<0:
            trans_land[i, 0]=0
        if trans_land[i, 1]<0:
            trans_land[i, 1]=0
        if trans_land[i, 0]>width:
            trans_land[i, 0]=width
        if trans_land[i, 1]>height:
            trans_land[i, 1]=height
    return trans_land
    # return landmark

def rotate_landmark(landmark, rotate_mat, height, width):
    rotate_land = np.zeros(np.shape(landmark))
    for i in range(np.shape(landmark)[0]):
        rotate_land[i] = np.around(np.array([landmark[i,0], landmark[i,1], 1]).dot(rotate_mat.T))
    rotate_land = np.array(rotate_land, 'int32')
    for i in range(rotate_land.shape[0]):
        if rotate_land[i, 0]<0:
            rotate_land[i, 0]=0
        if rotate_land[i, 1]<0:
            rotate_land[i, 1]=0
        if rotate_land[i, 0]>width:
            rotate_land[i, 0]=width
        if rotate_land[i, 1]>height:
            rotate_land[i, 1]=height
    return rotate_land

def get_rotate_img(img, landmark, first_frame_distance, first_frame_center, idx):
    height, width = np.shape(img)[:2]

    center_left_eye = [np.average(landmark[36:42, 0]), np.average(landmark[36:42, 1])]
    center_right_eye = [np.average(landmark[42:48, 0]), np.average(landmark[42:48, 1])]
    center_eyes = [(center_left_eye[0]+center_right_eye[0])/2, (center_left_eye[1]+center_right_eye[1])/2]
    distance = ((center_eyes[0]-center_right_eye[0])**2 + (center_eyes[1]-center_right_eye[1])**2)**(1/2)
    # print(center_left_eye, center_right_eye)
    # print(center_eyes)
    # print(distance)

    center = (center_eyes[0], center_eyes[1])

    radius = np.arctan((center_right_eye[1]-center_eyes[1])/(center_right_eye[0]-center_eyes[0]))
    degree = np.degrees(radius)
    # print(radius)
    # print(degree)
    if first_frame_distance is None:
        scale = 1
    else :
        scale = first_frame_distance / distance
    rotate_mat = cv2.getRotationMatrix2D(center, degree, scale)
    rotate_img = cv2.warpAffine(img, rotate_mat, (width, height))
    # cv2.imwrite("/WORKSPACE/VIDEO/ff++/original/actor/c23/align_original/01__exit_phone_room/%d_rotate.jpg" % idx, rotate_img)

    if first_frame_center is not None:
        trans_mat = np.float32([[1,0,first_frame_center[0]-center[0]],[0,1,first_frame_center[1]-center[1]]])
        trans_img = cv2.warpAffine(rotate_img, trans_mat, (width, height))
    else:
        trans_img = img
    # cv2.imwrite("/WORKSPACE/VIDEO/ff++/original/actor/c23/align_original/01__exit_phone_room/%d_trans.jpg" % idx, trans_img)

    face_land = rotate_landmark(landmark, rotate_mat, height, width)
    if first_frame_center is not None:
        face_land = trans_landmark(face_land, trans_mat, height, width)
    # print(rotate_land)

    return trans_img, distance, center, face_land

def calculate_bounding_landmark(landmark, start_point, bound_w, bound_h):
    bounding_land = np.zeros(landmark.shape)
    for i in range(landmark.shape[0]):
        bounding_land[i] = [landmark[i,0]-start_point[1], landmark[i,1]-start_point[0]]
        if bounding_land[i,0] < 0:
            bounding_land[i,0] = 0
        if bounding_land[i,0] > bound_w:
            bounding_land[i,0] = bound_w
        if bounding_land[i,1] < 0:
            bounding_land[i,1] = 0
        if bounding_land[i,1] > bound_h:
            bounding_land[i,1] = bound_h
    # print(bounding_land)
    bounding_land = np.array(bounding_land, 'int32')
    return bounding_land

def get_bounding_image(img, landmark):
    height, width, channels = np.shape(img)
    bound_w = max(landmark[:, 0]) - min(landmark[:, 0])
    bound_h = max(landmark[:, 1]) - min(landmark[:, 1])
    bound_w = int(bound_w)
    bound_h = int(bound_h)
    # print(bound_w, bound_h)

    st = [min(landmark[:, 1]), min(landmark[:, 0])]
    # print(st)
    bound_img = np.zeros((bound_h, bound_w, channels))
    # for h in range(bound_h):
    #     for w in range(bound_w):
    #         for c in range(channels):
    #             bound_img[h][w][c] = img[st[0]+h][st[1]+w][c]
    # st = [min(landmark[:, 0]), min(landmark[:, 1])]
    # # print(st)
    bound_img = img[st[0]:st[0]+bound_h, st[1]:st[1]+bound_w, :]
    # cv2.imwrite("bound.jpg", bound_img)

    bounding_land = calculate_bounding_landmark(landmark, st, bound_w, bound_h)

    return bound_img, bounding_land

def remove_background(img, landmark):
    height, width, channels = np.shape(img)
    mask = np.zeros((height, width))

    # remove background
    OUT_POINT = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,78,74,79,73,72,80,71,70,69,68,76,75,77]
    outline = np.array(landmark[OUT_POINT])
    # print(outline)
    cv2.fillPoly(mask, [outline], (255, 255, 255))

    # remove eye area
    eyeline = np.array(landmark[36:42])
    cv2.fillPoly(mask, [eyeline], (0, 0, 0))
    eyeline = np.array(landmark[42:48])
    cv2.fillPoly(mask, [eyeline], (0, 0, 0))
    # cv2.imwrite("mask.jpg", mask)
    # print(sssssssssssssss)

    nbg_img = np.zeros((height, width, channels))
    for h in range(height):
        for w in range(width):
            if mask[h][w]==0:
                continue
            nbg_img[h][w] = img[h][w]
    # cv2.imwrite("nbg.jpg", nbg_img)
    return nbg_img

def preprocess_img(img, detector, predictor, first_frame_distance, first_frame_center, idx):

    # stime = time.time()
    landmark = get_landmark(img, detector, predictor)
    # print(landmark)
    if landmark is None:
        return None, None, None, None
    # print("     landmark        ", time.time()-stime)
    # cv2.imwrite("/WORKSPACE/VIDEO/ff++/original/actor/c23/align_original/01__exit_phone_room/%d_face.jpg" % idx, img)

    # stime = time.time()
    rotate_img, distance, center, rotate_land = get_rotate_img(img, landmark, first_frame_distance, first_frame_center, idx)
    # print(distance, center)
    # print("     rotate image    ", time.time()-stime)

    # stime = time.time()
    bound_img, bounding_land = get_bounding_image(rotate_img, rotate_land)
    # print(bound_img)
    # cv2.imwrite("/WORKSPACE/VIDEO/ff++/original/actor/c23/align_original/01__exit_phone_room/%d_bound.jpg" % idx, bound_img)
    # print("     bounding box    ", time.time()-stime)

    # stime = time.time()
    no_bg_img = remove_background(bound_img, bounding_land)
    # no_bg_img = remove_background(rotate_img, rotate_land)
    # cv2.imwrite("/WORKSPACE/VIDEO/dfdc_data/part_02/no_bg_img.jpg", no_bg_img)
    # print("     remove bg       ", time.time()-stime)

    return no_bg_img, distance, center, landmark

def preprocess_video(video_file_name, detector, predictor, mtcnn, vidpath, facepath):
    print("Processing "+video_file_name+":")

    video_file_path = vidpath
    face_align_dir = facepath

    margin = (20, 20, 20, 20) # (start_point_x & y, end_point_x & y)
    batch_size = 30

    first_frame_distance = None
    video = cv2.VideoCapture(video_file_path)
    face_imgs = []
    idx = 0

    stime = time.time()
    frames = []
    while True:
        # ttime = time.time()
        success, frame = video.read()
        if not success:
            break
        height, width = frame.shape[:2]
        frames.append(frame)
        idx += 1
        if idx > 300:
            break
        if idx % batch_size == 0:
            first_frame_center = None
            batch_boxes, _ = mtcnn.detect(frames)
            pos = 0
            for boxes in batch_boxes:
                print("  %04d ... " % (idx - batch_size + pos), end='', flush=True)
                # print(box.shape)
                if boxes is None:
                    print("Done, there is no face")
                    continue
                st = [int(boxes[0,0]-margin[0]), int(boxes[0,1]-margin[1])]
                ed = [int(boxes[0,2]+margin[2]), int(boxes[0,3]+margin[3])]
                if st[0]<0:
                    st[0] = 0
                if st[1]<0:
                    st[1] = 0
                if ed[0]>width:
                    ed[0] = width
                if ed[1]>height:
                    ed[1] = height
                box = [st[0], st[1], ed[0], ed[1]]
                face_img = frames[pos][st[1]:ed[1], st[0]:ed[0], :]
                cv2.imwrite(face_align_dir + ("%04d_face.jpg" % (idx - batch_size + pos)), face_img)

                pre_img, distance, center, landmark = preprocess_img(face_img, detector, predictor, first_frame_distance, first_frame_center, (idx - batch_size + pos))
                if first_frame_distance is None:
                    first_frame_distance = distance
                if first_frame_center is None:
                    first_frame_center = center
                if pre_img is not None:
                    print("Done, saving ... ", end='', flush=True)
                    cv2.imwrite(face_align_dir + ("%04d.jpg" % (idx - batch_size + pos)), pre_img)
                    # cv2.imwrite(face_align_dir + ("%04d.jpg" % (idx - batch_size + pos)), pre_img)
                    np.save(face_align_dir + ("%04d_landmark.npy" % (idx - batch_size + pos)), landmark)
                    print("Done")
                else :
                    print("Done, no face")

                # face_imgs.append(face_img)
                # print(idx)
                pos += 1
            frames = []
            # print(time.time() - ttime)
            # ttime = time.time()
            # idx == 0
            # break
    # print(np.shape(face_imgs))
    print("-- Using time:", time.time() - stime)