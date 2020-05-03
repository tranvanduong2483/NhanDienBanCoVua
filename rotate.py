import cv2
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


images = glob.glob('./train_data/*/*.jpg')
for f in images:
    file_name = os.path.basename(f)
    dir_name = os.path.dirname(f)
    img = cv2.imread(f)
    for i in range(1, 4):
        new_img = rotate(img, i*90)
        new_name = '{}_{}{}'.format(file_name[0],(i*90),file_name[1:])
        new_path = os.path.join(dir_name, new_name)
        cv2.imwrite(new_path, new_img)