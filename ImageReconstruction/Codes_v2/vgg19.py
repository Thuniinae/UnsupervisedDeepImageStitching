import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import time

def Vgg19_simple(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    print(reuse)
    with tf.compat.v1.variable_scope("VGG19", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        #assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = Input(bgr.get_shape().as_list(), name='input')
        """ conv1 """
        network = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')(net_in)
        network = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')(network)
        """ conv2 """
        network = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')(network)
        network = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')(network)
        #conv_low = network
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')(network)
        """ conv3 """
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')(network)
        conv_low = tl.models.Model(inputs=net_in, outputs=network)
        network = Conv2d( n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')(network)
        """ conv4 """
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')(network)
        #conv = network
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')(network)  # (batch_size, 14, 14, 512)
        #conv = network
        """ conv5 """
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')(network)
        conv_high =tl.models.Model(inputs=net_in, outputs=network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')(network)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')(network)  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = Flatten(name='flatten')(network)
        network = Dense(n_units=4096, act=tf.nn.relu, name='fc6')(network)
        network = Dense(n_units=4096, act=tf.nn.relu, name='fc7')(network)
        network = Dense(n_units=1000, act=tf.identity, name='fc8')(network)
        print("build model finished: %fs" % (time.time() - start_time))
        return  conv_high, conv_low
