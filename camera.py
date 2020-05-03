import cv2


def get_frames():
    capture = cv2.VideoCapture(0)


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