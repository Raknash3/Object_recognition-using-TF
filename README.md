# Object_recognition-using-TF
Python code using tensorflow to recognize objects
Object database is obtained from fashion MNIST dataset that has around 60k images for training and 10k images for testing
Each image is 28*28 in size
There are 10 different object in the collection
Three layers 256 nodes in first layer and 128 nodes in second layer, 64 in third Activation func: Relu
Output layer nodes=10, activation func: Softmaxx
Optimizer=adam, loss function: sparse categorical cross entropy, performance metrics: accuracy
Train accuracy= 92.76% Test accuracy= 89.27%
