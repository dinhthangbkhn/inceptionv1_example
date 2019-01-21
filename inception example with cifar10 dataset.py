import h5py
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.layers import  Input, Dense, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding, Conv2D, MaxPooling2D, concatenate, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10



# Set up GPU 
# import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from tensorflow.keras import backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)


#Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255.
x_test = x_test/255.
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)


#code inception block (example)
def inception_block1(prev_layer):
    output_1 = Conv2D(5,(1,1), padding="same", activation='relu')(prev_layer)
    
    output_2_ = Conv2D(10,(1,1), padding="same", activation='relu')(prev_layer)
    output_2 = Conv2D(15,(3,3), padding="same", activation='relu')(output_2_)
    
    output_3 = Conv2D(2, (1,1), padding="same", activation='relu')(prev_layer)
    output_3 = Conv2D(4, (5,5), padding="same", activation='relu')(output_3)
    
    output_4 = Conv2D(4,(1,1), padding="same", activation='relu')(prev_layer)
    
    output = concatenate([output_1, output_2, output_3, output_4], axis = 3)
    
    return output


def inception_block2(prev_layer):
    output_1 = Conv2D(10,(1,1), padding="same", activation='relu')(prev_layer)
    
    output_2 = Conv2D(20,(1,1), padding="same", activation='relu')(prev_layer)
    output_2 = Conv2D(25,(3,3), padding="same", activation='relu')(output_2)
    
    output_3 = Conv2D(4, (1,1), padding="same", activation='relu')(prev_layer)
    output_3 = Conv2D(8, (5,5), padding="same", activation='relu')(output_3)
    
    output_4 = Conv2D(8,(1,1), padding="same", activation='relu')(prev_layer)
    
    output = concatenate([output_1, output_2, output_3, output_4], axis = 3)
    
    return output


def inception_block3(prev_layer):
    output_1 = Conv2D(20,(1,1), padding="same", activation='relu')(prev_layer)
    
    output_2 = Conv2D(40,(1,1), padding="same", activation='relu')(prev_layer)
    output_2 = Conv2D(50,(3,3), padding="same", activation='relu')(output_2)
    
    output_3 = Conv2D(8, (1,1), padding="same", activation='relu')(prev_layer)
    output_3 = Conv2D(15, (5,5), padding="same", activation='relu')(output_3)
    
    output_4 = Conv2D(15,(1,1), padding="same", activation='relu')(prev_layer)
    
    output = concatenate([output_1, output_2, output_3, output_4], axis = 3)
    return output


#build model 
def build_model():
    tf.reset_default_graph()
    K.clear_session()
    image_input = Input((32,32,3))
    output = inception_block1(image_input)
    output = MaxPooling2D(3, strides = 2)(output)
    output = inception_block2(output)
    output = MaxPooling2D(3, strides = 2)(output)
    output = inception_block3(output)
    output = AveragePooling2D(7, 1)(output)
    output = Flatten()(output)
    output = Dense(10, activation = "softmax")(output)
    model = Model(image_input, output)
    return model
    

model = build_model()
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

model.fit(x_train, y_train, validation_data = (x_test, y_test),  epochs = 20, batch_size = 512, shuffle = True)

