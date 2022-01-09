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

mrcnnpath = "../"
sys.path.append(mrcnnpath)
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import load_image_gt
import multiprocessing as mp

import re


class PUDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, images_dir, filename_prot):
        with open(filename_prot) as filin:
            prot_name = filin.readlines()
        
        self.IMG_SIZE = 512
        # define one class
        self.add_class("dataset", 1, "PU")
        # define data locations
        # find all images
        for image_id,prot in enumerate(prot_name):
            prot = prot.strip()
            img_path = images_dir + prot  + "/file_proba_contact.mat"
            annot_path = images_dir + prot + "/Peeling/Peeling.log"
            annot_path = images_dir + prot + "/" + prot + ".out"
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

    def get_sword_domain(self, output_file):
        domain = []
        regex = re.compile("([\d+\-\d+]+)")
        with open(output_file) as filin:
            for line in filin:
                match = regex.findall(line)
                if match:
                    for item in match:
                        if "-" in item:
                            domain.append(item.split("-"))
        all_domain = []
        try:
            first_pos = int(domain[0][0])
        except:
            return []
        beg = 1
        for PU in domain[:-1]:
            if PU[0] != "" and PU[1] != "":
                rel = np.loadtxt(f"{output_file[:-4]}.num")
                inf = np.where(rel >= int(PU[0]))[0] + 1
                sup = np.where(rel <= int(PU[1]))[0] + 1
                cov = np.in1d(inf,sup)
                res1 = inf[cov][0]
                res2 = inf[cov][-1]
                all_domain.append([res1,res2])

        return all_domain
    
    
    def get_PU(self, image_id):
        info = self.image_info[image_id]
        path = info['annot']
#         PU = self.log_to_res(path)
        PU = self.get_sword_domain(path)
        return PU
        
    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        # print(path, flush=True)
        # 0 1
        image = np.loadtxt(path,encoding='utf-8')
        image = np.expand_dims(image , axis = 2)
        pad = np.full((self.IMG_SIZE,self.IMG_SIZE,1), dtype = np.float32, fill_value= -1.)
        pad[:image.shape[0],:image.shape[0]] = image

        return pad
    
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        path = info['annot']
        image = self.load_image(image_id)
        # load BOX
        PU_list = self.get_PU(image_id)
        # create one array for all masks, each on a different channel
        masks = np.zeros([self.IMG_SIZE, self.IMG_SIZE, len(PU_list)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(PU_list)):
            box = PU_list[i]
            x, y = box[0], box[1]
            masks[x:y, x:y, i] = 1
            class_ids.append(1)
        return masks, np.asarray(class_ids, dtype='int32')

    def get_PU_score(self, rois, image):
       score = []
       for i in range(len(rois)):
           y1, x1, y2, x2 = rois[i]
           B1 = (image[:y1+1,x1:x2][:,:,0]).sum() *2
           B2 = (image[y1:y2,x2:][:,:,0]).sum() *2
           A = image[y1:y2,x1:x2][:,:,0].sum()
           score.append((A-(B1+B2)) / (A+(B1+B2)))
       return score
    
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
        return np.mean(results)

    def compute_iou(self, image_id):

        
        config = PUConfig()
        # tf.keras.backend.reset_uids() 
        model = MaskRCNN(mode='inference', model_dir='../results/', config = config )
        # load weights (mscoco) and exclude the output layers
        now = datetime.datetime.now()

        model.load_weights("../results/sword_resize_heads20211121T1353/mask_rcnn_sword_resize_heads_0080.h5", by_name=True)
        print(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(self, config, image_id)
        # image = self.load_image(image_id)
        r = model.detect([image])
        y_pred = r[0]["rois"]
        PU = self.get_PU(image_id)
        y_true = []

        for c in PU:
            y_true.append([c[0],c[0],c[1],c[1]])
        y_true = np.array(y_true)

        y_pred = np.array(y_pred)

        return self.calculate_iou(y_true, y_pred)


    
    def perf(self, model, directory, epoch):
        all_id = self.image_ids
        if len(all_id) < 5000:
            dataset = "test"
        else:
            dataset = "train"

        PU_scores_predict =  []
        iou = []
        PU_scores_true = []
        for image_id in all_id:
            y_true = []

            start = time.time()
            print(image_id)
            true_shape = self.load_image(image_id).shape[0]
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
               load_image_gt(self, model.config, image_id)
            r = model.detect([image])[0]
            y_pred = r["rois"]
            
            # Predict score
            PU_scores_predict += self.get_PU_score(y_pred, image)
            lab = self.get_PU(image_id)
           

            for i_box in range(len(lab)):
                x1, y1 = lab[i_box]
                x1, y1 = np.array(lab[i_box]) * model.config.to_dict()["IMAGE_MAX_DIM"] // true_shape
                y_true.append([x1, x1, y1, y1])

            PU_scores_true += self.get_PU_score(y_true, image)
            iou.append(self.calculate_iou(np.array(y_true), y_pred))
            print("TIME : ", time.time() - start)

        np.save(f"../results/{directory}/predict_scores_{dataset}_{epoch}.npy", np.array(PU_scores_predict))
        np.save(f"../results/{directory}/true_scores_{dataset}_{epoch}.npy", np.array(PU_scores_true))
        np.save(f"../results/{directory}/iou_{dataset}_{epoch}.npy", np.array(iou))

        
            

# define a configuration for the model
class PUConfig(Config):
    # define the name of the configuration
    NAME = "real_pad_512prot_1048"
    # number of classes (background + PU)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    # STEPS_PER_EPOCH = 131
    STEPS_PER_EPOCH = 1048
    # # MAX_GT_INSTANCES = 50
    # # POST_NMS_ROIS_INFERENCE = 500
    # # POST_NMS_ROIS_TRAINING = 1000
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 512
    # TRAIN_ROIS_PER_IMAGE = 200
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.5
    TOP_DOWN_PYRAMID_SIZE = 512

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (256, 256)  # (height, width) of the mini-mask

    BACKBONE = "resnet50"

    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = IMAGE_MAX_DIM = 512

    # IMAGE_MIN_SCALE = 0

    MEAN_PIXEL = np.array([0.5])
    IMAGE_CHANNEL_COUNT = 1

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

    def txt_to_config(self, file):
        cfg = {}
        with open(file) as filin:
            for line in filin:
                items = line.strip().split(",",maxsplit=1)
                key = items[0].strip()[2:-1]
                value = items[1].strip()[:-1]
                locals()[key] = value
                cfg[key] = value
        # self.NAME = "NOOOMM"


    def from_dict(self, config_dict):
        for key in config_dict:
            self.key = config_dict[key]

def main():

    train_txt = "../data/train_set_512_con.txt"
    test_txt = "../data/test_set_512.txt"
    val_txt = "../data/val_set_512.txt"
    sword_dir = "../data/data_sword/"

    train_set = PUDataset()
    train_set.load_dataset(sword_dir, train_txt)
    train_set.prepare()

    val_set = PUDataset()
    val_set.load_dataset(sword_dir, val_txt)
    val_set.prepare()


    test_set = PUDataset()
    test_set.load_dataset(sword_dir, test_txt)
    test_set.prepare()

    config = PUConfig()

    config.display()
    print(len(train_set.image_ids))
    print(len(val_set.image_ids))

    # # INFERENCE or TRAINING

    model = MaskRCNN(mode="training", model_dir='../results/', config = config )
    # model = MaskRCNN(mode="inference", model_dir='../results/', config = config )
    
    # folder = "real_pad_256prot_13120211224T1251"
    # model.load_weights(f'../results/{folder}/mask_rcnn_real_pad_256prot_131_0050.h5', by_name=True) 
    now = datetime.datetime.now()
    model.keras_model.summary()
    # # # train weights (output layers or 'heads')
    model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=50,  layers='all')

    config.to_txt(now)
 
    # test_set.perf(model,folder, "50")
    # train_set.perf(model,folder, "50")


    # print("TIME : ", time.time() - start)

if __name__ == "__main__": 
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()
