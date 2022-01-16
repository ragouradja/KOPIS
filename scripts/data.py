from mask_rcnn import *

import time
from joblib import Parallel, delayed
import numpy as np

def check(dataset, ID):
    image = dataset.load_image(ID)
    if image.shape[0] <= 512:
        print(ID)
        name = dataset.image_info[ID]["path"].split("/")[3] + "\n"
        return name
    return ""
train_txt = "../data/train_set.txt"
test_txt = "../data/test_set.txt"
val_txt = "../data/val_set.txt"
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


liste = Parallel(n_jobs= 8, verbose = 0, prefer="threads")(delayed(check)
(train_set, ID) for ID in train_set.image_ids)

f = open("../data/train_set_512.txt","w")
f.writelines(liste)
f.close()




liste = Parallel(n_jobs= 8, verbose = 0, prefer="threads")(delayed(check)
(test_set, ID) for ID in test_set.image_ids)

f = open("../data/test_set_512.txt","w")
f.writelines(liste)
f.close()



liste = Parallel(n_jobs= 8, verbose = 0, prefer="threads")(delayed(check)
(val_set, ID) for ID in val_set.image_ids)

f = open("../data/val_set_512.txt","w")
f.writelines(liste)
f.close()


