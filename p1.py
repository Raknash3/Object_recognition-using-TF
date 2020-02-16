from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
a=keras.datasets.fashion_mnist #load the data to variable a
(img_train,label_train),(img_test,label_test)=a.load_data() #create train and test set
class_name=['T-shirt/top','Trouser','Pullover','dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle_Boot'] #since class names are not provided in the dataset we had to include it manually
#use len() and .dim() to measure the size of each variable.
#normalizartion
img_train= img_train/255.0
img_test= img_test/255.0
#create a model
model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), #flatten layer
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),# three hidden layer with 128 nodes and relu as activation function
        keras.layers.Dense(10,activation='softmax')# final layer with ten nodes coressponding to the labels
        ])
#specify optimiser, loss function and metric to track
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model
model.fit(img_train,label_train,epochs=15) #default batch size=32
#evaluate the accuracy
test_loss,test_acc= model.evaluate(img_test,label_test)
print(test_acc)
#make predictions
p=model.predict(img_test)


#np.argmax(p[0])
##### To display a data####################################
#def display_img(s):
#    plt.imshow(s,cmap=plt.cm.binary)
#    plt.show()
    
#x_1=x_train[0]
#display_img(x_1)
################################################

