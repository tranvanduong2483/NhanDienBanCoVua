import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_board_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    ret, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] # superfluous dimensions
    children = []

    for i in range(len(contours)):
        epsilon = 0.1*cv2.arcLength(contours[i],True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 500:
            children.append(hierarchy[i])

    children = np.array(children)
    values,counts = np.unique(children[:,3], return_counts=True)
    contour = contours[values[np.argmax(counts)]]
    epsilon = 0.1*cv2.arcLength(contour,True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_frames():
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    if not capture.isOpened():
        raise RuntimeError('Failed to start camera.')

    while True:
        ret, img = capture.read()
        cv2.imshow('webcam', img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    capture.release()

    return img


img = get_frames()
approx = find_board_contour(img)