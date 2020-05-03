from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob



model = load_model('./model_50x50.hd5')

name_map = {
    'K0': 'Vua trắng',
    'Q0': 'Hậu trắng',
    'R0': 'Xe trắng',
    'B0': 'Tượng trắng',
    'N0': 'Mã trắng',
    'P0': 'Tốt trắng',
    '__': 'Khoảng trống',
    'k1': 'Vua đen',
    'q1': 'Hậu đen',
    'r1': 'Xe đen',
    'b1': 'Tượng đen',
    'n1': 'Mã đen',
    'p1': 'Tốt đen',
}


while True:
    images = glob.glob('./train_data/*/*.jpg')

    index = np.random.randint(len(images))
    img = cv2.imread(images[index])

    pred = predict(img, model, plot=False)
    print('Dự đoán là   :', name_map[pred])
    print('Đúng phải là :', name_map[images[index][13:15]])
    print(' ')


    plt.imshow(img)
    plt.show()
