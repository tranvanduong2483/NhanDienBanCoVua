import time

import cv2
import numpy as np

import os.path
from os import path


# Cắt hình ảnh
def correct_perspective(img, approx):
    pts = approx.reshape(4, 2)
    tl, tr, br, bl = (
        pts[np.argmin(np.sum(pts, axis=1))],
        pts[np.argmin(np.diff(pts, axis=1))],
        pts[np.argmax(np.sum(pts, axis=1))],
        pts[np.argmax(np.diff(pts, axis=1))]
    )

    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    src = np.array([tl, tr, br, bl], dtype='float32')
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (int(w), int(h)))

    return cv2.resize(img, (400, 400))


# Lấy từng ô
def get_square(img, row, col):
    width = img.shape[0]
    square = width // 8
    x1, y1 = row * square, col * square
    x2, y2 = x1 + square, y1 + square
    return img[x1:x2, y1:y2]


# Tìm đường viền
def find_board_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # superfluous dimensions
    children = []

    for i in range(len(contours)):
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 500:
            children.append(hierarchy[i])

    children = np.array(children)
    values, counts = np.unique(children[:, 3], return_counts=True)
    contour = contours[values[np.argmax(counts)]]
    epsilon = 0.1 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


train_path = os.getcwd() + '/train_data'
label_map = list('QRBNP_kqrbnp')
# K

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError('Failed to start camera.')

quit = False
ret, img = capture.read()

for piece in label_map:
    if quit: break
    path = os.path.join(train_path, piece)
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(8 * 8):
        if quit: break

        while True:
            try:
                ret, img = capture.read()
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                board = correct_perspective(img, approx)
                col = i % 8
                row = i // 8
                square = get_square(board, row, col)

                key_press = cv2.waitKey(5)
                if key_press & 0xFF == ord(' '):
                    file_name = '{}_{}.jpg'.format(piece, time.time())
                    cv2.imwrite(os.path.join(path, file_name), square)
                    break

                if key_press & 0xFF == ord('q'):
                    quit = True
                    break
                if key_press & 0xFF == ord('r'):
                    approx = find_board_contour(img)
                    continue

                width = board.shape[0]
                s = width // 8
                x1, y1 = col * s, row * s
                x2, y2 = x1 + s, y1 + s
                cv2.rectangle(board, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.imshow('Capturing piece: {}'.format(piece), board)
            except:
                continue

cv2.destroyAllWindows()
capture.release()
