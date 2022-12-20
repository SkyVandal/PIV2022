import scipy
import numpy as np
import cv2 as cv


def split_video_to_frames(path):
    vc = cv.VideoCapture(path)
    c = 1

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        img_nr = str(c)
        n_zeros = (4 - len(img_nr)) * '0'
        cv.imwrite('frames/rgb_' + n_zeros + img_nr + '.jpg', frame)
        print()
        c = c + 1
        cv.waitKey(1)
    vc.release()


def get_keypoints_ORB(image_path):
    img = cv.imread(image_path, 0)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    return kp, des


def read_keypoints_descriptors(path):
    data = scipy.io.loadmat(path)
    p = data['p']  # (2,N) numpy array, where N is the total number of keypoints
    d = data['d']  # (128,N) numpy array

    return d, p
