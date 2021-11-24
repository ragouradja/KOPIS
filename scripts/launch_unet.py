import argparse
import os
import tensorflow as tf
import sys
import multiprocessing as mp
import mask_rcnn


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True 
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    session = tf.compat.v1.Session(config=config)
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    print(tf.__version__)
    with tf.device('/gpu:0'):
        mask_rcnn.main()


    # 379s 100row 1 cpu
    # 293s 100 row 5 cpu
    # 277s




if __name__ == "__main__":
    main()