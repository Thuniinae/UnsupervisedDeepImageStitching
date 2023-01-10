import tensorflow as tf
import numpy as np
import cv2 
import os
import glob

from models import reconstruction
from utils import DataLoader, load, save
import constant

tf.compat.v1.disable_eager_execution()

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-200000'
batch_size = constant.TEST_BATCH_SIZE



# define dataset
with tf.compat.v1.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.compat.v1.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))


# define testing generator function
with tf.compat.v1.variable_scope('Reconstruction', reuse=None):
    print('testing = {}'.format(tf.compat.v1.get_variable_scope().name))
    lr_test_stitched, hr_test_stitched = reconstruction(test_inputs)
 


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.compat.v1.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder)

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
        length = len(glob.glob(os.path.join(test_folder, 'warp1/*.jpg')))
        out_path = "../results_Multi/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_image_clips(i), axis=0)
            

            #stich warp1,warp2
            _, stitch_result = sess.run([lr_test_stitched, hr_test_stitched], feed_dict={test_inputs: input_clip[...,0:6]})
        
            
            stitch_result = (stitch_result+1) * 127.5    
            stitch_result = stitch_result[0]
            #stitch_result = cv2.resize(stitch_result,(2443,2154))

            name = data_loader['warp1']['frame'][i].split('/')[-1]
            path = out_path + name
            cv2.imwrite(path, stitch_result)
            print('i = {} / {}'.format( i, length))
            
        print("===================DONE!==================")  

    inference_func(snapshot_dir)

    

