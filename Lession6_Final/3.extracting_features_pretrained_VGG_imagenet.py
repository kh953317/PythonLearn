# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:45:34 2020

@author: Lee
"""

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import progressbar #$ pip install progressbar
#import argparse
import random
# import the necessary packages for HD5
import h5py
import os
# trained model on extracted features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

# define the constructor
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
                             "exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", outputPath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0
    # add method used to add data to our buffer
    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
    # flush method
    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}
    # store the raw string names of the class labels in a separate dataset
    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        #dt = h5py.special_dtype(vlen = unicode)
        # In Python 3, unicode has been renamed to str.
        dt = h5py.special_dtype(vlen = str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels
    # function close will be used to write any data left in the buffers to HDF5 as
    # well as close the dataset
    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
###############################################################################
def rank5_accuracy(preds, labels):
    # initialize the rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0
    # loop over the predictions and ground-truth labels
    for (p, gt) in zip(preds, labels):
        # sort the probabilities by their index in descending
        # order so that the more confident guesses are at the
        # front of the list
        p = np.argsort(p)[::-1]
        
        # check if the ground-truth label is in the top-5
        # predictions
        if gt in p[:5]:
            rank5 += 1
            # check to see if the ground-truth is the #1 prediction
        if gt == p[0]:
            rank1 += 1
        # compute the final rank-1 and rank-5 accuracies
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))
    # return a tuple of the rank-1 and rank-5 accuracies
    return (rank1, rank5)
###############################################################################
# feature extraction
path_dataset = "D:/2020/Teaching/Books/Programs/datasets/17flowers"
# batch size of images to be passed through network
bs = 32 
# grab the list of images that we’ll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images(path_dataset))
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the
# labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)


buffer_size_default=1000
#path_output = "D:/2020/Teaching/Books/Programs/datasets/hdf5_2/animals_features.hdf5"
#path_output = "D:/2020/Teaching/Books/Programs/datasets/hdf5_2/caltech_101_features.hdf5"
path_output_features = "D:/2020/Teaching/Books/Programs/datasets/hdf5/17flowers_features6.hdf5"
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),
path_output_features, dataKey="features", bufSize=buffer_size_default)
dataset.storeClassLabels(le.classes_)
# initialize the progress bar
widgets = ["Extracting Features: ", 
           progressbar.Percentage(), " ",
           progressbar.Bar(), " ", 
           progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
    
    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)
    # pass the images through the network and use the outputs as
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    # reshape the features so that each image is represented by
    # a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and labels to our HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)
# close the dataset
dataset.close()
pbar.finish()
###############################################################################
# Training a Classifier on Extracted Features
p = path_output_features
#p = "E:/Teaching/Books/Programs/datasets/hdf5_2/17flowers_features.hdf5"

db = h5py.File(p, "r")
i = int(db["labels"].shape[0] * 0.75)
#use 75% of the data for training and 25% for evaluation, we can simply
#compute the 75% index i into the database. Any data before the index i is considered training data
#– anything after i is testing data

#define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
#parameter C, the strictness of the Logistic Regression
#classifier
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
#model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)
H = model.fit(db["features"][:i], db["labels"][:i])
#model.summary()
#H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=100,verbose=1)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

# serialize the model to disk
print("[INFO] saving model...")
#path_output_model = "D:/2020/Teaching/Books/Programs/datasets/hdf5_2/animals.cpickle"
#path_output_model = "E:/Teaching/Books/Programs/datasets/hdf5_2/caltech_101.cpickle"
path_output_model = "D:/2020/Teaching/Books/Programs/datasets/hdf5/17flowers.cpickle"
#f = open(args["model"], "wb")
f = open(path_output_model, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

###############################################################################
# rank 1 and rank 5
#p_model = "D:/2020/Teaching/Books/Programs/datasets/hdf5_2/animals.cpickle"
p_model = path_output_model
model = pickle.loads(open(p_model, "rb").read())
# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk 
#p_db = "D:/2020/Teaching/Books/Programs/datasets/hdf5_2/animals_features.hdf5"
p_db = path_output_features
db = h5py.File(p_db, "r")
i = int(db["labels"].shape[0] * 0.8)

# make predictions on the testing set then compute the rank-1
# and rank-5 accuracies
print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

# display the rank-1 and rank-5 accuracies
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))

print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

# close the database
db.close()




