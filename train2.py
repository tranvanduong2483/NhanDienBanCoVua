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
from livelossplot import PlotLossesKeras


def training():
    label_map = list(['K0', 'Q0', 'R0', 'B0', 'N0', 'P0', '__', 'k1', 'q1', 'r1', 'b1', 'n1', 'p1'])
    num_classes = len(label_map)

    img_size = (50,50)

    images = glob.glob('./train_data/*/*.jpg')

    data_rows = []

    for f in images:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE) #Chuyển ảnh về trắng đen
        img = cv2.resize(img, img_size).flatten() #Chuyển về màn 1 chiều các mỗi ô chứa độ sáng (2500 ô)

        label = f[13:15] # lấy label của từ đường dẫn file ảnh
        index = label_map.index(label)


        data_rows.append(np.insert(img, 0, index))

    train_data = pd.DataFrame(data_rows)

    #train_data.to_csv('./train_data/train_data.csv', index=False)

    #train_data = pd.read_csv('./train_data/train_data.csv',skiprows=1, header=None)

    input_shape = (*img_size, 1)

    X = train_data.drop([0], axis='columns').values     # Mảng các ảnh ô cờ
    y = to_categorical(train_data[0].values)            # Khởi tạo Mảng % mỗi ô cờ với dự đoán

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = X_train.reshape(X_train.shape[0], *img_size, 1)
    X_val = X_val.reshape(X_val.shape[0], *img_size, 1)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    X_train = X_train/ 255
    X_val = X_val/ 255

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


    # 4. Train the model
    hist = model.fit(X_train, y_train,
                     batch_size=50,
                     epochs=1000,
                )

    model.save('./model_50x50.hd5')
    print('Accuracy:', hist.history)
training()

