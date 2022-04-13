# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:03:24 2020

@author: Lee
"""


## import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# for ImageToArrayPreprocessor
from tensorflow.keras.preprocessing.image import img_to_array
# for AspectAwarePreprocessor and SimpleDatasetLoader
import imutils
import cv2
import os
import numpy as np
# for myVGGNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers.core import Activation
from tensorflow.keras.layers.core import Flatten
from tensorflow.keras.layers.core import Dropout
from tensorflow.keras.layers.core import Dense
from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
# for ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###############################################################################
class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the the Keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)
###############################################################################
class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0
        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired dimension
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0) 
        
        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # to crop along the width
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
        # now that our images have been resized, we need to
        # re-grab the width and height, followed by performing the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]
        # finally, resize the image to the provided spatial
        # dimensions to ensure our output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
###############################################################################
class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))
###############################################################################
class MyVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            input_shape=inputShape
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same"
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same"
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same"
        ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
##############################################################################

# path to input dataset
path_dataset = r"E:\Python\ASE_classification_data\20210611-1"
#path_dataset = "D:/2020/Teaching/Books/Programs/datasets/caltech_101"
#path_dataset = "D:\\2020\\Deep Learning\\NEU surface defect database\\NEU surface defect database\\NEU surface defect database"
print("[INFO] loading images...")
imagePaths = [ x for x in list(paths.list_images(path_dataset)) if x.split(os.path.sep)[-2] !='jpg']
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths ]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
image_width = 64
image_height = 64
aap = AspectAwarePreprocessor(image_width, image_height)
#aap = AspectAwarePreprocessor(200, 200) # for NEU database
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data,labels) = sdl.load(imagePaths,verbose = 100)
data = data.astype('float') / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.80,random_state  =42)

y_test_org = testY

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
# To train our flower classifier we’ll be using the MiniVGGNet architecture along with the SGD optimizer
# initialize the optimizer and model
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, 
                         width_shift_range=0.1,
                         height_shift_range=0.1, 
                         shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, 
                         fill_mode="nearest")
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)

model = MyVGGNet.build(width = image_width,
                       height = image_height,
                       depth =3,
                       classes = len(classNames))

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()
# train the network
print("[INFO] training network...")
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
num_epochs = 50
H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=32),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // 32, epochs=num_epochs,
        verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))
# plot the training loss and accuracy
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# 計算『混淆矩陣』(Confusion Matrix)，顯示測試集分類的正確及錯認總和數
import pandas as pd 
predictions = model.predict_classes(testX) 
pd.crosstab(y_test_org, predictions, rownames=['實際值'], colnames=['預測值'])