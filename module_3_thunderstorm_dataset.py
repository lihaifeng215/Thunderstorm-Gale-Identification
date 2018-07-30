import numpy as np
'''
1. 将上一步标准化后的数据添加一个关于风速的标签，从而构建标准的雷暴大风数据集
2. 将大风和非大风按照一定比例混合，并添加风速标签，构造雷暴大风标准数据集
3. 输入：上一步标准化化后的雷达区域图像
4. 输出： 雷暴大风数据集
'''

'''
append label to each record for classification
'''
def append_regression_label(origin_data):
    append_reg_label = np.column_stack((origin_data, origin_data[:, 4]))
    print("append_regression_label.shape:", append_reg_label.shape)
    print("max wind speed:", np.max(append_reg_label[:, 4]))
    print("min wind speed:", np.min(append_reg_label[:, 4]))
    print("median wind speed:", np.median(append_reg_label[:, 4]))
    return append_reg_label

'''
merge_positive_negative_sample
'''
def merge_sample(positive_sample, negative_sample):
    merged_sample = np.row_stack((positive_sample, negative_sample))
    # 按时间排序
    merged_sample_sort = merged_sample[np.lexsort(merged_sample[:, ::-1].T)]
    return merged_sample_sort

'''
get_balanced_dataset
'''
def get_balanced_dataset(dataset, balance_factor=1, wind_speed=15):
    print("dataset.shape:", dataset.shape)
    big_wind = dataset[(dataset[:, 4] >= wind_speed) & (np.max(dataset[:,5:5+13*13*9], axis=1) >= 30)]
    print("big_wind.shape:", big_wind.shape)
    small_wind = dataset[(dataset[:, 4] < wind_speed) & (np.max(dataset[:,5:5+13*13*9], axis=1) <= 40)]
    print("small_wind.shape:", small_wind.shape)
    np.random.shuffle(small_wind)
    small_wind = small_wind[-big_wind.shape[0] * balance_factor:, :]
    balanced_dataset = np.row_stack((big_wind, small_wind))
    np.random.shuffle(balanced_dataset)
    # 按时间进行排序
    # balanced_dataset_sort = balanced_dataset[np.lexsort(balanced_dataset[:, ::-1].T)]
    # print(balanced_dataset_sort)
    # print("balanced_dataset_sort.shape:", balanced_dataset_sort.shape)
    # return balanced_dataset_sort
    return balanced_dataset

'''
split training data and testing data
'''
def split_train_and_test(dataset, test_number):
    test_start = -test_number
    train = dataset[:test_start, :]
    test = dataset[test_start:, :]
    return train, test

if __name__ == '__main__':
    # 使用13 x 13 大小
    positive_data = np.loadtxt("file/positive_sample_17y_Radar_denoised_13x13.csv", delimiter=',')
    print("positive_data.shape:", positive_data.shape)
    negative_data = np.loadtxt("file/negative_sample_05_17y_Radar_denoised_13x13.csv", delimiter=',')
    print("negative_data.shape:", negative_data.shape)
    merged_sample_sort = merge_sample(positive_data, negative_data)
    print("merged_sample_sort.shape",merged_sample_sort.shape)

    # 使用平衡的训练集和测试集
    balanced_dataset = get_balanced_dataset(merged_sample_sort,balance_factor=4, wind_speed=15)
    train_dataset, test_dataset = split_train_and_test(balanced_dataset, 6000)
    train_dataset = append_regression_label(train_dataset)
    test_dataset = append_regression_label(test_dataset)
    print("train_dataset.shape:",train_dataset.shape)
    print("test_dataset.shape:", test_dataset.shape)
    np.savetxt("file/train_test/train_dataset_17y_Radar_denoised_13x13.csv", train_dataset, delimiter=',', fmt='%f')
    np.savetxt("file/train_test/test_dataset_17y_Radar_denoised_13x13.csv", test_dataset, delimiter=',', fmt='%f')
