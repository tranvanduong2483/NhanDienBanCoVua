import cv2
import numpy as np
from keras.models import load_model


# Khởi tạo các biến toàn cục
def init():
    global approx, model, name_map, label_map, capture

    approx = np.array([[[97, 189]],
                       [[353, 196]],
                       [[356, 453]],
                       [[91, 456]]], dtype=np.int32)
    name_map = {
        'V ': 'Vua trắng',
        'H ': 'Hậu trắng',
        'X ': 'Xe trắng',
        'T ': 'Tượng trắng',
        'M ': 'Mã trắng',
        'TT': 'Tốt trắng',
        '--': 'Khoảng trống',
        'v ': 'Vua đen',
        'h ': 'Hậu đen',
        'x ': 'Xe đen',
        't ': 'Tượng đen',
        'm ': 'Mã đen',
        'tt': 'Tốt đen',
    }
    label_map = list(['V ', 'H ', 'X ', 'T ', 'M ', 'TT', '--', 'v ', 'h ', 'x ', 't ', 'm ', 'tt'])
    model = load_model('./model_50x50.hd5')

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError('Failed to start camera.')


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


def XacDinhKhungBanCo(img):
    global approx
    try:
        approx = find_board_contour(img)

        # Draw contour and have a look at the result.

        contour_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

        cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 3)

        print([approx])
        return contour_img

        board = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

        board = correct_perspective(board, approx)

        return board

    except:
        print("ERROR")
        return img


# Dự đoán quân cờ một ô vuông
def predict(img, model, img_size=(50, 50)):
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.reshape(1, *img_size, 1) / 255
    pred = model.predict(img)

    return label_map[np.argmax(pred)]


# Dự đoán các quân trên cờ toàn bộ bàn cờ
def predict_all(board, print=False):
    predict_array = [
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
    ]
    for row in range(8):
        for col in range(8):
            square = get_square(board, row, col)
            predict_chess = predict(square, model)
            predict_array[row][col] = predict_chess

            if predict_chess != '--':
                width = board.shape[0]
                s = width // 8
                x1, y1 = col * s, row * s
                x2, y2 = x1 + s, y1 + s
                cv2.rectangle(board, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(board, predict_chess, ((x1 + x2) // 2 - 6, (y1 + y2) // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)



    return [predict_array, board]




# Hiển thị mảng hai chiều
def show_chess_board_teminal(_2d_array):
    row = len(_2d_array)
    col = len(_2d_array[0])

    print('|-------------------------------|')
    print('|            Result             |')
    print('|-------------------------------|')
    for i in range(row):
        print('|', end=' ')
        for j in range(col):
            if j == col-1:
                print(_2d_array[i][j], end='')
            else:
                print(_2d_array[i][j], end = '  ')
        print('|')
    print('|-------------------------------|')

    print()

def listener_short_cut(img):
    global approx
    key_press = cv2.waitKey(5)
    if key_press & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        capture.release()
        exit()
    if key_press & 0xFF == ord('r'):
        approx = find_board_contour(img)


init()
while True:

    # ẢNH ĐẦU VÀO
    ret, img = capture.read()                           # Chụp ảnh từ camera
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)      # Xoay ảnh 90 độ để thuận màn hình, dễ quan sát
    board = correct_perspective(img, approx)            # Cắt ra bàn cờ vua


    # NHẬN DIỆN
    predict_array, changed_board = predict_all(board)   # Lấy kết quả dự đoán

    # HIỂN THỊ KẾT QUẢ
    show_chess_board_teminal(predict_array)             # Hiển thị mảng được dự đoán trên terminal
    cv2.imshow('Result', changed_board)                 # Hiển thị ảnh bàn cờ đã dự đoán

    # CHỜ SỰ KIỆN NGƯỜI DÙNG
    #listener_short_cut(img)                                # Lắng nghe các sự kiện điều khiển của người dùng

    key_press = cv2.waitKey(5)
    if key_press & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        capture.release()
        exit()
    if key_press & 0xFF == ord('r'):
        approx = find_board_contour(img)