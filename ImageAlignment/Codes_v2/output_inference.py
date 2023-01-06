import tensorflow as tf
import os
import numpy as np
import cv2
import glob

from models import H_estimator, output_H_estimator
from utils import DataLoader, load, save
import constant
import skimage


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-1000000'
batch_size = constant.TEST_BATCH_SIZE

# define dataset
with tf.compat.v1.name_scope('dataset'):
    ##########testing###############
    tf.compat.v1.disable_eager_execution()
    test_inputs = tf.compat.v1.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    test_size = tf.compat.v1.placeholder(shape=[batch_size, 2, 1], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))
    print('test size = {}'.format(test_size))



with tf.compat.v1.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.compat.v1.get_variable_scope().name))
    test_coarsealignment = output_H_estimator(test_inputs, test_size, False)
    


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.compat.v1.Session(config=config) as sess:


    # initialize weights
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.compat.v1.global_variables()]
    loader = tf.compat.v1.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        
        print("------------------------------------------")
        print("generating aligned images for training set")
        # dataset
        data_loader = DataLoader(train_folder)
        length = len(glob.glob(os.path.join(train_folder, 'input1/*.jpg')))
             #create folder if not exist
        out_path = "../output/training"
        if not os.path.exists(out_path+"/mask1"):
            os.makedirs(out_path+"/mask1")
        if not os.path.exists(out_path+"/mask2"):
            os.makedirs(out_path+"/mask2")
        if not os.path.exists(out_path+"/warp1"):
            os.makedirs(out_path+"/warp1")
        if not os.path.exists(out_path+"/warp2"):
            os.makedirs(out_path+"/warp2")
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i, None, None), axis=0)
            size_clip = np.expand_dims(data_loader.get_size_clips(i), axis=0)
            
            coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})
            
            coarsealignment = coarsealignment[0]
            warp1 = (coarsealignment[...,0:3]+1.)*127.5
            warp2 = (coarsealignment[...,3:6]+1.)*127.5
            mask1 = coarsealignment[...,6:9] * 255
            mask2 = coarsealignment[...,9:12] * 255
            
            path1 = out_path+'/warp1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path1, warp1)
            path2 = out_path+'/warp2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path2, warp2)
            path3 = out_path+'/mask1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path3, mask1)
            path4 = out_path+'/mask2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path4, mask2)
                   
            print('i = {} / {}'.format(i+1, length))

        print("-----------training set done--------------")
        print("------------------------------------------")
        
        print()
        print()
        
        print("------------------------------------------")
        print("generating aligned images for testing set")
        # dataset
        data_loader = DataLoader(test_folder)
        length = len(glob.glob(os.path.join(test_folder, 'input1/*.jpg')))
        #create folder if not exist
        out_path = "../output/testing"
        if not os.path.exists(out_path+"/mask1"):
            os.makedirs(out_path+"/mask1")
        if not os.path.exists(out_path+"/mask2"):
            os.makedirs(out_path+"/mask2")
        if not os.path.exists(out_path+"/warp1"):
            os.makedirs(out_path+"/warp1")
        if not os.path.exists(out_path+"/warp2"):
            os.makedirs(out_path+"/warp2")
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i, None, None), axis=0)
            size_clip = np.expand_dims(data_loader.get_size_clips(i), axis=0)
            
            coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})
            
            coarsealignment = coarsealignment[0]
            warp1 = (coarsealignment[...,0:3]+1.)*127.5
            warp2 = (coarsealignment[...,3:6]+1.)*127.5
            mask1 = coarsealignment[...,6:9] * 255
            mask2 = coarsealignment[...,9:12] * 255
            
            path1 = out_path+'/warp1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path1, warp1)
            path2 = out_path+'/warp2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path2, warp2)
            path3 = out_path+'/mask1/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path3, mask1)
            path4 = out_path+'/mask2/' + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path4, mask2)
                     
            print('i = {} / {}'.format(i+1, length))

        print("-----------testing set done--------------")
        print("------------------------------------------")

     
    inference_func(snapshot_dir)



