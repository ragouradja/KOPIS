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
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import concatenate, UpSampling2D

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import MeanIoU
import math
import cv2

from joblib import Parallel, delayed
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean


import datetime
import time
# LOSS VERY HIGH WITH THIS PATH !
# mrcnnpath = r"../../Mask_RCNN/"
# sys.path.append(mrcnnpath)

from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import load_image_gt
    

class PUDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, images_dir, prot_name):
        self.model = None
        self.IMG_SIZE = 256
        # define one class
        self.add_class("dataset", 1, "PU")
        # define data locations
        # find all images
        for image_id,prot in enumerate(prot_name):
            prot = prot[0]
            img_path = images_dir + prot + f"/PDBs_Clean/{prot}.pdb{prot[-1]}" + "/file_proba_contact.mat"
            annot_path = images_dir + prot + f"/PDBs_Clean/{prot}.pdb{prot[-1]}" "/Peeling/Peeling.log"
            self.add_image('dataset', image_id=image_id, path=img_path, annot = annot_path)

    def log_to_res(self, logfile):
        all_coords = []
        with open(logfile, encoding='utf-8') as filin:
            for line in filin:
                if not line.startswith("#"):
                    clean_coord = line.strip().split()[5:]
                    xy = [[int(clean_coord[x]),int(clean_coord[x+1])] for x in range(0,len(clean_coord)-1,2)]
                    all_coords.append(xy)
        return all_coords[-1]
    
    
    def get_PU(self, image_id):
        info = self.image_info[image_id]
        path = info['annot']
        PU = self.log_to_res(path)
        PU_list = []
        labels = []
        shape_image = sorted(PU)[-1][-1]
        for couple in PU:
            PU_list = []
            for res in couple:
                # PU_list.append(int(res * self.IMG_SIZE /shape_image )) # With resizing of images
                PU_list.append(res) 
            labels.append(PU_list)
        return labels
        
    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        # print(path, flush=True)
        image = np.loadtxt(path,encoding='utf-8') * 255
        image = cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.uint8)
        # image = resize(image, (self.IMG_SIZE, self.IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
        return image
    
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        path = info['annot']
        image = self.load_image(image_id)
        # load BOX
        PU_list = self.get_PU(image_id)
        # create one array for all masks, each on a different channel
        masks = np.zeros([image.shape[0], image.shape[0], len(PU_list)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(PU_list)):
            box = PU_list[i]
            x, y = box[0], box[1]
            masks[x:y, x:y, i] = 1
            class_ids.append(1)
        return masks, np.asarray(class_ids, dtype='int32')


    def set_model(self, model):
        self.model = model

    def calculate_iou(self, y_true, y_pred):
        results = []


        y_true = np.array(sorted(y_true.tolist()))
        y_pred = np.array(sorted(y_pred.tolist()))
            
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)


        for i in range(0,y_true.shape[0]):
            results_PU = []

            # boxTrue
            x_boxTrue_tleft = y_true[i,0]  # numpy index selection
            y_boxTrue_tleft = y_true[i,1]
            boxTrue_width = y_true[i,2]
            boxTrue_height = y_true[i,3]
            area_boxTrue = (boxTrue_width * boxTrue_height)

            for j in range(y_pred.shape[0]):

                # boxPred
                x_boxPred_tleft = y_pred[j,0]
                y_boxPred_tleft = y_pred[j,1]
                boxPred_width = y_pred[j,2]
                boxPred_height = y_pred[j,3]
                area_boxPred = (boxPred_width * boxPred_height)


                # calculate the bottom right coordinates for boxTrue and boxPred

                # boxTrue
                x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
                y_boxTrue_br = y_boxTrue_tleft + boxTrue_height # Version 2 revision

                # boxPred
                x_boxPred_br = x_boxPred_tleft + boxPred_width
                y_boxPred_br = y_boxPred_tleft + boxPred_height # Version 2 revision


                # calculate the top left and bottom right coordinates for the intersection box, boxInt

                # boxInt - top left coords
                x_boxInt_tleft = np.max([x_boxTrue_tleft,x_boxPred_tleft])
                y_boxInt_tleft = np.max([y_boxTrue_tleft,y_boxPred_tleft]) # Version 2 revision

                # boxInt - bottom right coords
                x_boxInt_br = np.min([x_boxTrue_br,x_boxPred_br])
                y_boxInt_br = np.min([y_boxTrue_br,y_boxPred_br]) 

                # Calculate the area of boxInt, i.e. the area of the intersection 
                # between boxTrue and boxPred.
                # The np.max() function forces the intersection area to 0 if the boxes don't overlap.


                # Version 2 revision
                area_of_intersection = \
                np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

                iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)


                # This must match the type used in py_func
                iou = iou.astype(np.float32)
                #print(f"BOX {i} MASK {j} IOU {iou}")
                # append the result to a list at the end of each loop
                results_PU.append(iou)

            results.append(max(results_PU))
        # return the mean IoU score for the batch
        print(np.mean(results))
        return np.mean(results)

    def compute_iou(self, image_id):
        print(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(self, self.model.config, image_id)
        print(image.shape)
        # image = self.load_image(image_id)
        r = self.model.detect([image])
        y_pred = r[0]["rois"]
        PU = self.get_PU(image_id)
        y_true = []

        for c in PU:
            y_true.append([c[0],c[0],c[1],c[1]])
        y_true = np.array(y_true)

        y_pred = np.array(y_pred)
        return self.calculate_iou(y_true, y_pred)

    def iou_parallel(self):
        start = time.time()
        all_iou = Parallel(n_jobs = 2, verbose = 0, prefer="threads")(delayed(self.compute_iou)
            (image_id) for image_id in self.image_ids)

        print(all_iou)
        print(np.mean(all_iou))
        print(time.time() - start)


# define a configuration for the model
class PUConfig(Config):
    # define the name of the configuration
    NAME = "sword_pad"
    # number of classes (background + PU)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131
    # # MAX_GT_INSTANCES = 50
    # # POST_NMS_ROIS_INFERENCE = 500
    # # POST_NMS_ROIS_TRAINING = 1000
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # TRAIN_ROIS_PER_IMAGE = 200
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.5
    TOP_DOWN_PYRAMID_SIZE = 256

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    BACKBONE = "resnet50"

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # IMAGE_MIN_SCALE = 0

 
    IMAGE_CHANNEL_COUNT = 3
    LEARNING_RATE = 0.0001

    def to_txt(self, now):
        config_dict = self.to_dict().items()
        folder = "{}{:%Y%m%dT%H%M}".format(self.NAME.lower(),now)
        folder_results = "../results/"
        filename = "{}{}{:%Y%m%dT%H%M}.cfg".format(folder_results, self.NAME.lower(),now)
        f = open(filename, "w")
        for x in config_dict:
            f.write(f"{x}\n")
        f.close()

        os.system("move {} {}{}".format(filename, folder_results, folder))
        print("Config in {} folder".format(folder))

def main():
    start = time.time()
    # images_dir = "../sword/SWORD/PDBs_Clean/"
    images_dir = "../data/data_sword/"

    prot_name = next(os.walk(images_dir))[1]

    prot_name_dtf = pd.DataFrame(prot_name)
    prot_train = prot_name_dtf.sample(frac= 0.7, random_state = 42)
    prot_test = prot_name_dtf[~prot_name_dtf.isin(prot_train)].dropna()

    train_set = PUDataset()
    train_set.load_dataset(images_dir, prot_train.values.tolist())
    train_set.prepare()

    test_set = PUDataset()
    test_set.load_dataset(images_dir, prot_test.values.tolist())
    test_set.prepare()

    config = PUConfig()

    config.display()
    print(train_set.image_ids)

    # INFERENCE or TRAINING
    with tf.device("/gpu:0"):
        model = MaskRCNN(mode='inference', model_dir='../results/', config = config )
    # load weights (mscoco) and exclude the output layers
    now = datetime.datetime.now()

    model.load_weights("../results/sword_resize_heads20211121T1353/mask_rcnn_sword_resize_heads_0080.h5", by_name=True)

    # train weights (output layers or 'heads')
    # model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5,  layers='heads')
    # print(model.keras_model.summary())
    # tf.keras.utils.plot_model(model, to_file="mrcnn.png",show_shapes=True, show_layer_names = True)
    # config.to_txt(now)
    start = time.time()
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(test_set, model.config, 1)
    model.detect([image])
    print("TIME : ", time.time() - start)
    
    # test_set.set_model(model)
    # test_set.iou_parallel()    

if __name__ == "__main__":
    main()