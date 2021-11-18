import tensorflow as tf
import os
import sys

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import os
import sys
import copy
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import concatenate, UpSampling2D

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import time
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import MeanIoU
import math
import cv2

from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
# import the necessary packages
from collections import namedtuple

# define the `Detection` object for IOU(
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
from PIL import Image, ImageOps
# importing XML parsing library for parsing the data
import xml.etree.ElementTree as ET

def log_to_res(peeling_file_name):
    all_coords = []
    with open(peeling_file_name) as filin:
        for line in filin:
            if not line.startswith("#"):
                clean_coord = line.strip().split()[5:]
                xy = [[int(clean_coord[x]),int(clean_coord[x+1])] for x in range(0,len(clean_coord)-1,2)]
                all_coords.append(xy)
        return all_coords


# Define IoU metric
def compute_iou(y_true, y_pred):
    iou = []
    for i in range(len(y_true)):
        
        xA = max(y_pred[i][0], y_true[i][0])
        yA = max(y_pred[i][1], y_true[i][1])
        xB = min(y_pred[i][2], y_true[i][2])
        yB = max(y_pred[i][3], y_true[i][3])
        interArea = (yB-yA)*(xB-xA)
        boxAArea = (y_pred[i][2] - y_pred[i][0] + 1) * (y_pred[i][3] - y_pred[i][1] + 1)
        boxBArea = (y_true[i][2] - y_true[i][0] + 1) * (y_true[i][3] - y_true[i][1] + 1)

        iou.append(interArea/float(boxAArea+boxBArea-interArea))
    return np.mean(iou)

def do_train_parallel(train_prot, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, image_directory):
    # 141 sans parallel pour train

    # Get and resize train images and masks
    X_train = []
    Y_train = []


    start = time.time()
    all_train = Parallel(n_jobs= 7, verbose = 0, prefer="processes")(delayed(get_train)
    (image_directory,IMG_HEIGHT, IMG_WIDTH, X_train, Y_train, prot) for prot in train_prot.values.tolist())
    print("TIME : ",time.time() - start)
    print(len(all_train))
    X_train = np.zeros((len(train_prot), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.float32) 
    Y_train = np.zeros((len(train_prot), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    for n, img in enumerate(all_train):
        X_train[n] = img[0][0]
        Y_train[n] = img[1][0]

    return X_train, Y_train

def do_test_parallel(test_prot, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, image_directory):
    # 141 sans parallel pour test

    # Get and resize test images and masks
    X_test = []
    Y_test = []
    sizes_test = []

    start = time.time()
    all_test = Parallel(n_jobs= 7, verbose = 0, prefer="processes")(delayed(get_test)
    (image_directory,IMG_HEIGHT, IMG_WIDTH, X_test, Y_test, prot,sizes_test) for prot in test_prot.values.tolist())
    print("TIME : ",time.time() - start)

    X_test = np.zeros((len(test_prot), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.float32)    
    Y_test = np.zeros((len(test_prot), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

    for n, img in enumerate(all_test):
        X_test[n] = img[0][0]
        Y_test[n] = img[1][0]
        sizes_test.append(img[2][0])
        print(img[2][0])
    return X_test, Y_test, sizes_test


def get_train(image_directory,IMG_HEIGHT, IMG_WIDTH,X_train, Y_train, prot):
    prot=prot[0]
    path = image_directory + prot + "/"

    img = np.loadtxt(path + prot + '.mat')
    img = img.reshape(img.shape[0],img.shape[0],1)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train.append(img)
    
    img = np.loadtxt(path + prot + '_PU.mat') 
    img = img.reshape(img.shape[0],img.shape[0],1)        
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # img[img > 0] = 1 
    Y_train.append(img)
    return X_train, Y_train

def get_test(image_directory,IMG_HEIGHT, IMG_WIDTH,X_test, Y_test, prot, sizes_test):
    prot=prot[0]
    path = image_directory + prot + "/"

    img = np.loadtxt(path + prot + '.mat')
    img = img.reshape(img.shape[0],img.shape[0],1)
    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test.append(img)
    
    img = np.loadtxt(path + prot + '_PU.mat') 
    img = img.reshape(img.shape[0],img.shape[0],1)        
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # img[img > 0] = 1 
    Y_test.append(img)
    return X_test, Y_test, sizes_test

def do_data_label_parallel():
        # 141 sans parallel pour test

    # Get and resize test images and masks
    X_test = []
    Y_test = []
    sizes_test = []

    start = time.time()
    all_test = Parallel(n_jobs= 7, verbose = 0, prefer="processes")(delayed(get_test)
    (image_directory,IMG_WIDTH, IMG_WIDTH, X_test, Y_test, prot,sizes_test) for prot in test_prot.values.tolist())
    print("TIME : ",time.time() - start)

    X_test = np.zeros((len(test_prot), IMG_WIDTH, IMG_WIDTH, 3), dtype = np.float32)    
    Y_test = np.zeros((len(test_prot), IMG_WIDTH, IMG_WIDTH, 3), dtype = np.uint8)

    for n, img in enumerate(all_test):
        X_test[n] = img[0][0]
        Y_test[n] = img[1][0]
        sizes_test.append(img[2][0])
        print(img[2][0])
    return X_test, Y_test, sizes_test


def get_data_label(prot_name, IMG_WIDTH, prot):

    labels = []
    data = []
    PU_scaled = []
    PU_list = []

    PU = log_to_res(f"../../data_peeling/{prot}/Peeling.log")[-1][0]
    image = np.loadtxt(f"../../data_peeling/{prot}/full_prob_map.mat")
    for res in PU:
        PU_list.append(int(res * IMG_WIDTH / image.shape[0]))
        PU_list.append(int(res * IMG_WIDTH / image.shape[0]))
    PU_scaled.append(PU_list)
    labels.append(PU_list)
    image = resize(image, (IMG_WIDTH, IMG_WIDTH), mode='constant', preserve_range=True)
    image = cv2.cvtColor(np.array(image).astype(np.float32), cv2.COLOR_RGB2BGR)
    data.append(image)

    data = np.array(data).astype(np.float32)
    labels = np.array(labels).astype(np.float32)



def main():
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 1

    directory = "../../data_peeling/"
    image_directory = "../data/"

    prot_name = next(os.walk(directory))[1]


    labels = []
    data = []
    start = time.time()
    for prot in prot_name:
        # PU_scaled = []
        PU_list = []

        PU = log_to_res(f"../../data_peeling/{prot}/Peeling.log")[-1][0]
        image = np.loadtxt(f"../../data_peeling/{prot}/full_prob_map.mat")
        for res in PU:
            PU_list.append((res * IMG_WIDTH / image.shape[0]) / IMG_WIDTH)
            # PU_list.append((res * IMG_WIDTH / image.shape[0]) / IMG_WIDTH)
        # PU_scaled.append(PU_list)
        labels.append(PU_list)
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        image = cv2.cvtColor(np.array(image).astype(np.float32), cv2.COLOR_RGB2BGR)
        data.append(image)

    print(time.time() - start)
    data = np.array(data).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    
    print(x_train)
    print(y_train)
    # base_model1 = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH,IMG_WIDTH, 3))
    # chopped_resnet1 = Model(inputs=[base_model1.input], outputs=[base_model1.layers[90].output])
    # localization_output1 = Flatten()(chopped_resnet1.output)
    # localization_output1 = Dense(units=4, activation='sigmoid')(localization_output1)
    # model = Model(inputs=[chopped_resnet1.input], outputs=[localization_output1])
    # model.summary()

    from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Input
    from tensorflow.python.keras.applications.xception import Xception  

    inp = Input(shape=(IMG_WIDTH, IMG_WIDTH, 3))
    base_model = ResNet50(include_top=False, input_tensor=inp, weights='imagenet')
    y = base_model.layers[90].output
    y = GlobalAveragePooling2D()(y)
    y = Dense(2, activation='sigmoid')(y)
    model = Model(inputs=inp, outputs=y)
    model.compile(optimizer='adam', metrics=["accuracy"],loss=['mse'])
    # model.compile(optimizer='adam', metrics=["accuracy"],loss=['mse'], run_eagerly = True)

    model.summary()

    # model.compile(optimizer='adam', metrics=["accuracy"],loss=['mse'], run_eagerly = True)

    # # Build U-Net model
    # inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    # # s = Lambda(lambda x: x / 255) (inputs)

    # c1 = Conv2D(16, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    # c1 = Dropout(0.4) (c1)
    # c1 = Conv2D(16, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    # p1 = MaxPooling2D((2, 2)) (c1)

    # c2 = Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    # c2 = Dropout(0.4) (c2)
    # c2 = Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    # p2 = MaxPooling2D((2, 2)) (c2)

    # c3 = Conv2D(64, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    # c3 = Dropout(0.4) (c3)
    # c3 = Conv2D(64, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    # p3 = MaxPooling2D(pool_size=(2, 2)) (c3)

    # c4 = Conv2D(128, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    # c4 = Dropout(0.4) (c4)
    # c4 = Conv2D(128, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    # p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    # c5 = Conv2D(256, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    # c5 = Dropout(0.4) (c5)
    # c5 = Conv2D(256, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    # u6 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same') (c5)
    # u6 = concatenate([u6, c4])
    # c6 = Conv2D(128, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    # c6 = Dropout(0.4) (c6)
    # c6 = Conv2D(128, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    # u7 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same') (c6)
    # u7 = concatenate([u7, c3])
    # c7 = Conv2D(64, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    # c7 = Dropout(0.4) (c7)
    # c7 = Conv2D(64, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    # u8 = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same') (c7)
    # u8 = concatenate([u8, c2])
    # c8 = Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    # c8 = Dropout(0.4) (c8)
    # c8 = Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    # u9 = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same') (c8)
    # u9 = concatenate([u9, c1], axis=3)
    # c9 = Conv2D(16, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    # c9 = Dropout(0.4) (c9)
    # c9 = Conv2D(16, (4, 4), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    # c9 = Flatten()(c9)
    # # outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    # outputs = Dense(4, activation='relu')(c9)
    
    # model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss='mse', metrics="accuracy",run_eagerly=True)
    # model.summary()


    
    # Fit model
    # earlystopper = EarlyStopping(patience=1, verbose=1,monitor='val_accuracy')
    # checkpointer = ModelCheckpoint('model_unet.h5', verbose=1, save_best_only=True,monitor='val_accuracy')
    results = model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=5)
    model.save("reset_2points.h5")
    pred_test = model.predict(x_test)
    print("VRAI BOX : ",y_test[0])
    print("Predicted box :",pred_test[0])
    
    # preds_test_t = (pred_test > 0.5).astype(np.uint8)
    # # Create list of upsampled test masks
    # preds_test_upsampled = []
    # for i in range(len(pred_test)):
    #     preds_test_upsampled.append(resize(np.squeeze(pred_test[i]), 
    #                                     (sizes_test[i][0], sizes_test[i][1]), 
    #                                     mode='constant', preserve_range=True))

    # print(train_prot)
    
    # plt.imshow(X_test[2])
    # plt.show()

    # plt.imshow(preds_test_upsampled[2])
    # plt.show()

    # plt.imshow(preds_test_t[2])

    # plt.show()

if __name__ == "__main__":
    main()