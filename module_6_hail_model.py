import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,f1_score

model = Sequential()
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid',input_shape=(7,7,1)))
model.add(Activation('elu'))
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid'))
model.add(Activation('elu'))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy')

data = np.loadtxt("file1/dataset.txt",delimiter=',',dtype=int)
features = data[:,:-1].reshape(data.shape[0],1,7,7).transpose(0,2,3,1)
label = data[:,-1]

model.fit(features,label,batch_size=features.shape[0],epochs=2000,verbose=2,validation_split=0.5)
model.save("file1/models/model1_7x7.hdf5")
predict = model.predict(features,batch_size=1,verbose=0)
np.savetxt('file1/models/predict.txt',np.column_stack((predict,label)),delimiter=",",fmt='%f')


