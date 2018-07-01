
# coding: utf-8

# ## Train Network to Drive Car in Simulation

# ## Data Preprocessing

# In[6]:

import cv2
import keras
import tensorflow as tf
import csv
import numpy as np
import os 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D, Dense, Lambda, Cropping2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# SOME CONSTANTS
CENTER_IDX = 0
STEERING_ANGLE_IDX = 3


def extract_file_name(file):
    """
        Assume last backslash is start of the file name
    """
    assert(file is not None)
    name_start = file.rfind("/")
    return file[name_start+1:]


def read_input_dir(folder_name, raw_inputs):
    """
       read input from specific folder 
    """
    if not os.path.exists(folder_name):
        raise Exception('folder %s does not exist' % folder_name)
    with open(folder_name + "/driving_log.csv") as f:
        reader = csv.reader(f)
        for data in reader:
            # extract name 
            center_img = os.path.join(folder_name+"/IMG", extract_file_name(data[CENTER_IDX]))
            # append data 
            raw_inputs.append((center_img, float(data[STEERING_ANGLE_IDX])))
            

def data_generator(raw_inputs, batch_size=32):
    total_samples = len(raw_inputs)
    print(raw_inputs.shape)
    while 1:
        for offset in range(0, total_samples, batch_size):
            samples = raw_inputs[offset:offset+batch_size]
            inputs = []
            labels = []
            for sample in samples:
                if (not os.path.exists(sample[0])):
                    print("Failed %s " % sample[0])
                inputs.append(cv2.cvtColor(cv2.imread(sample[0]), cv2.COLOR_BGR2RGB))
                labels.append(sample[1])
            X_train = np.array(inputs)
            Y_train = np.array(labels)
            yield shuffle(X_train, Y_train)
        
def create_model():
    model = Sequential()
    # normalize input 
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0)))) 
    model.add(Conv2D(24, 5, 5,activation='relu', subsample=(2,2))) # input: 65x320x3, 31x158x3x16
    model.add(BatchNormalization())
    model.add(Dropout(p=0.2))
    model.add(Conv2D(36, 5, 5,activation='relu',subsample=(2,2))) # input: 31x158x3x32, 13x76x3x32
    model.add(BatchNormalization())
    model.add(Dropout(p=0.2))
    model.add(Conv2D(48, 5, 5,activation='relu',subsample=(2,2))) # input: 31x158x3x32, 13x76x3x32 
    model.add(BatchNormalization())
    model.add(Dropout(p=0.2))
    model.add(Conv2D(64, 3, 3,activation='relu')) # input: 31x158x3x32, 13x76x3x32 
    model.add(BatchNormalization())
    model.add(Dropout(p=0.2))
    model.add(Conv2D(64, 3, 3,activation='relu')) # input: 31x158x3x32, 13x76x3x32 
    model.add(Dropout(p=0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(p=0.4))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(p=0.4))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


## MAIN START HERE
if __name__ == "__main__":
    raw_inputs = []
    folder_list = [
        "../collect_data/drive_forward_data",
        "../collect_data/drive_reverse_data",
        "../collect_data/drive_left_data",
        "../collect_data/drive_right_data",
        "../collect_data2/side_drive_data_1",
        "../collect_data2/side_drive_data2",
        "../collect_data3/bridge_data",
        "../collect_data4",
        "../collect_data5"
    ]

    # Read all data path and steering angles to list
    for each_folder in folder_list:
        read_input_dir(each_folder, raw_inputs)
        print (len(raw_inputs))

    # convert to numpy array
    raw_inputs = np.array(raw_inputs)
    # split raw data to train and validation
    train_raw_inputs, validation_raw_inputs = train_test_split(raw_inputs, test_size=0.2)

    #create generators for train and validation
    train_generator = data_generator(train_raw_inputs)
    validation_generator = data_generator(validation_raw_inputs)

    # ready to train model
    # save model check point based on val_loss decreasing
    filepath="best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    # create and train model
    my_model = create_model()
    my_model.fit_generator(train_generator, samples_per_epoch=len(train_raw_inputs), callbacks=callback_list,nb_epoch=20, 
                   validation_data=validation_generator, nb_val_samples=len(validation_raw_inputs))






