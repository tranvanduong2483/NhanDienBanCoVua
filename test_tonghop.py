import glob
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2

def training():
    ds = ['K0', 'Q0', 'R0', 'B0', 'N0', 'P0', '__', 'k1', 'q1', 'r1', 'b1', 'n1', 'p1']
    label_map = list(ds)
    img_size = (50,50)

    num_classes = len(label_map)
    images = glob.glob('./train_data/*/*.jpg')
    data_rows = []

    for f in images:
        thumuc = f[13:15]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size).flatten()
        label = label_map.index(thumuc)
        data_rows.append(np.insert(img, 0, label))

    train_data = pd.DataFrame(data_rows)
    train_data.head()
    train_data.to_csv('./train_data/train_data.csv', index=False)

def creatModel():
    img_size = (50, 50)
    train_data = pd.read_csv('./train_data/train_data.csv', skiprows=1, header=None)
    input_shape = ((*img_size), 1)  # (50,50,1)
    X = train_data.drop([0], axis=1).values
    y = to_categorical(train_data[0].values)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], *img_size, 1)
    X_val = X_val.reshape(X_val.shape[0], *img_size, 1)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    X_train /= 255
    X_val /= 255

    num_classes = 13

    pool_size = (4, 4)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_normal',
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.save('./model_50x50.hd5')



model = load_model('./model_50x50.hd5')

name_map = {
    'K0': 'Vua trắng',
    'Q0': 'Hậu trắng',
    'R0': 'Xe trắng',
    'B0': 'Tượng trắng',
    'N0': 'Mã trắng',
    'P0': 'Tốt trắng',
    '__': 'Empty Square',
    'k1': 'Vua đen',
    'q1': 'Hậu đen',
    'r1': 'Xe đen',
    'b1': 'Tượng đen',
    'n1': 'Mã đen',
    'p1': 'Tốt đen',
}
def predict(img, model, img_size=(50, 50), plot=False):
    ds = ['K0', 'Q0', 'R0', 'B0', 'N0', 'P0', '__', 'k1', 'q1', 'r1', 'b1', 'n1', 'p1']
    label_map = list(ds)

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if plot:
        plt.imshow(img, cmap='gray')

    img = img.reshape(1, *img_size, 1) / 255
    pred = model.predict(img)

    print(pred*100)
    return label_map[np.argmax(pred)]


training()
creatModel()



img = cv2.imread('./train_data/__/__1586706323.666057.jpg')

pred = predict(img, model, plot=False)
print('The image is a', name_map[pred])
plt.imshow(img)
plt.show()
