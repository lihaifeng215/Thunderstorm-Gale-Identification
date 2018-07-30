import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid',input_shape=(-1,7,7,20)))
model.add(Activation('elu'))
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid'))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(1,activation='elu'))

model.compile(optimizer=Adam(lr=0.001),loss='mae')
print("model compiled!")

# get all_batch_data of data_13x13
def get_batch_13x13(data_13x13):
    all_batch_data = np.array(data_13x13[:,5:-1])
    all_batch_label = np.array(data_13x13[:,-1])
    all_batch_data = all_batch_data.reshape(all_batch_data.shape[0], 20, 13, 13)
    all_batch_data = np.transpose(all_batch_data, axes=(0,2,3,1))
    # choice channel
    all_batch_data = all_batch_data[:,:,:,:]
    all_batch_label = data_13x13[:, -1]

    return all_batch_data, all_batch_label

# get all_batch_data of data_7x7
def get_batch_7x7(data_7x7):
    all_batch_data = np.array(data_7x7[:, 5:-1])
    all_batch_label = np.array(data_7x7[:, -1])
    all_batch_data = all_batch_data.reshape(all_batch_data.shape[0],20, 7,7)
    all_batch_data = np.transpose(all_batch_data, axes=(0,2,3,1))

    # choice channel
    all_batch_data = all_batch_data[:,:,:,:]
    all_batch_label = data_7x7[:, -1]
    return all_batch_data, all_batch_label

# Normalize data
def preprocess(all_batch_data, size, channel):
    transform_data = all_batch_data.reshape(-1, channel)
    transform_data = MinMaxScaler().fit_transform(transform_data)
    all_batch_data = transform_data.reshape(all_batch_data.shape[0], size*2+1, size*2+1, channel)
    return all_batch_data


train_data_13x13 = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled.csv",delimiter=',')
print("train_data_13x13.shape:", train_data_13x13.shape)
test_data_13x13 = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled.csv",delimiter=',')
train_data_7x7 = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv",delimiter=',')
print("train_data_7x7.shape:", train_data_7x7.shape)
test_data_7x7 = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv",delimiter=',')

train_batch_data_13x13, train_batch_label_13x13 = get_batch_13x13(train_data_13x13)
print("train_batch_data_13x13, train_batch_label_13x13:", train_batch_data_13x13.shape,train_batch_label_13x13.shape)
test_batch_data_13x13, test_batch_label_13x13 = get_batch_13x13(test_data_13x13)
print("test_batch_data_13x13, test_batch_label_13x13:", test_batch_data_13x13.shape,test_batch_label_13x13.shape)

train_batch_data_7x7, train_batch_label_7x7 = get_batch_7x7(train_data_7x7)
print("train_batch_data_7x7, train_batch_label_7x7:", train_batch_data_7x7.shape, train_batch_label_7x7.shape)
test_batch_data_7x7, test_batch_label_7x7 = get_batch_7x7(test_data_7x7)
print("test_batch_data_7x7, test_batch_label_7x7:", test_batch_data_7x7.shape, test_batch_label_7x7.shape)

train_batch_data_13x13 = preprocess(train_batch_data_13x13, size=6, channel=20)
test_batch_data_13x13 = preprocess(test_batch_data_13x13, size=6, channel=20)
train_batch_data_7x7 = preprocess(train_batch_data_7x7, size=3, channel=20)
test_batch_data_7x7 = preprocess(test_batch_data_7x7, size=3, channel=20)

print(train_batch_data_13x13.shape, test_batch_label_13x13.shape, test_batch_data_13x13.shape, test_batch_label_13x13.shape)
print(train_batch_data_7x7.shape, test_batch_label_7x7.shape, test_batch_data_7x7.shape, test_batch_label_7x7.shape)

rand_ix = np.random.permutation(len(train_batch_data_13x13))
train_batch_data_13x13, train_batch_data_7x7, train_batch_label_13x13 = train_batch_data_13x13[rand_ix], train_batch_data_7x7[rand_ix], train_batch_label_13x13[rand_ix]

model.fit(train_batch_data_7x7,train_batch_label_7x7,batch_size=500,epochs=500,verbose=2,validation_split=0.2)
pred = model.predict(test_batch_data_7x7)
model.evaluate(test_batch_data_7x7,test_batch_label_7x7)