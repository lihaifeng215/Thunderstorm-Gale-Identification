import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
"""
1. 使用上一步生成的雷暴大风训练集训练深度学习模型
2. 输入：雷暴大风训练集和测试集
3. 输出：训练得到的模型，以及在测试集上的误差计算结果
"""
model = Sequential()
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid',input_shape=(13,13,9)))
model.add(Activation('elu'))
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(30,(3,3),strides=(1,1),padding='valid'))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1,activation='elu'))

model.compile(optimizer=Adam(lr=0.001),loss='mae')
print("model compiled!")

# get all_batch_data of data_13x13
def get_batch_13x13(data_13x13):
    all_batch_data = np.array(data_13x13[:,5:-1])
    all_batch_label = np.array(data_13x13[:,-1])
    all_batch_data = all_batch_data.reshape(all_batch_data.shape[0], 9, 13, 13)
    all_batch_data = np.transpose(all_batch_data, axes=(0,2,3,1))
    # choice channel
    all_batch_data = all_batch_data[:,:,:,:]
    all_batch_label = data_13x13[:, -1]

    return all_batch_data, all_batch_label

# Normalize data
def preprocess(all_batch_data, size, channel):
    transform_data = all_batch_data.reshape(-1, channel)
    transform_data = MinMaxScaler().fit_transform(transform_data)
    all_batch_data = transform_data.reshape(all_batch_data.shape[0], size*2+1, size*2+1, channel)
    return all_batch_data

if __name__ == "__main__":
    train_data_13x13 = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13.csv",delimiter=',')
    print("train_data_13x13.shape:", train_data_13x13.shape)
    test_data_13x13 = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13.csv",delimiter=',')

    train_batch_data_13x13, train_batch_label_13x13 = get_batch_13x13(train_data_13x13)
    print("train_batch_data_13x13, train_batch_label_13x13:", train_batch_data_13x13.shape,train_batch_label_13x13.shape)
    test_batch_data_13x13, test_batch_label_13x13 = get_batch_13x13(test_data_13x13)
    print("test_batch_data_13x13, test_batch_label_13x13:", test_batch_data_13x13.shape,test_batch_label_13x13.shape)

    train_batch_data_13x13 = preprocess(train_batch_data_13x13, size=6, channel=9)
    test_batch_data_13x13 = preprocess(test_batch_data_13x13, size=6, channel=9)


    print(train_batch_data_13x13.shape, test_batch_label_13x13.shape, test_batch_data_13x13.shape, test_batch_label_13x13.shape)


    model.fit(train_batch_data_13x13,train_batch_label_13x13,batch_size=1000,epochs=500,verbose=2,validation_split=0.1)
    pred = model.predict(test_batch_data_13x13)
    result = np.column_stack((test_batch_label_13x13, pred))
    np.savetxt("file/keras_13x13_result.csv",result,delimiter=',',fmt='%f')
    model.save("file/models/cnn_13x13.hdf5")

    from sklearn import metrics
    data = result
    r2_score = metrics.r2_score(data[:,0],data[:,1])
    print("r2_score:",r2_score)
    mae = metrics.mean_absolute_error(data[:,0],data[:,1])
    rmse = np.sqrt(metrics.mean_squared_error(data[:,0],data[:,1]))
    print("mae:",mae)
    print("rmse:",rmse)
