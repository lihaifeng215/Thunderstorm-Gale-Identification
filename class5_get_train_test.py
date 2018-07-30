import numpy as np
class get_train_test:
    def __init__(self):
        pass

    '''
    merge_positive_negative_sample
    '''
    def merge_sample(self, positive_sample, negative_sample):
        merged_sample = np.row_stack((positive_sample, negative_sample))
        # 按时间排序
        merged_sample_sort = merged_sample[np.lexsort(merged_sample[:, ::-1].T)]
        return merged_sample_sort

    '''
    get_balanced_dataset
    '''
    def get_balanced_dataset(self, dataset, balance_factor=1, wind_speed=15):
        print("dataset.shape:", dataset.shape)
        big_wind = dataset[(dataset[:, 4] >= wind_speed) & (np.max(dataset[:,5:5+7*7*9], axis=1) >= 30)]
        # big_wind = dataset[(dataset[:, 4] >= wind_speed)]
        print("big_wind.shape:", big_wind.shape)
        small_wind = dataset[(dataset[:, 4] < wind_speed) & (np.max(dataset[:,5:5+7*7*9], axis=1) <= 40)]
        # small_wind = dataset[(dataset[:, 4] < wind_speed)]
        print("small_wind.shape:", small_wind.shape)
        np.random.shuffle(small_wind)
        small_wind = small_wind[-big_wind.shape[0] * balance_factor:, :]
        balanced_dataset = np.row_stack((big_wind, small_wind))
        # np.random.shuffle(balanced_dataset)
        # 按时间进行排序
        balanced_dataset_sort = balanced_dataset[np.lexsort(balanced_dataset[:, ::-1].T)]
        print(balanced_dataset_sort)
        print("balanced_dataset_sort.shape:", balanced_dataset_sort.shape)
        return balanced_dataset_sort

    '''
    split training data and testing data
    '''
    def split_train_and_test(self, dataset, test_number):
        test_start = -test_number
        train = dataset[:test_start, :]
        test = dataset[test_start:, :]
        return train, test

if __name__ == '__main__':
    # 使用7 x 7 大小
    positive_sample = np.loadtxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv",delimiter=',')
    negative_sample = np.loadtxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv", delimiter=',')
    train_test = get_train_test()
    merged_sample_sort = train_test.merge_sample(positive_sample, negative_sample)
    print("merged_sample_sort.shape",merged_sample_sort.shape)

    # 使用平衡的训练集和测试集
    balanced_dataset = train_test.get_balanced_dataset(merged_sample_sort,balance_factor=1, wind_speed=15)
    train_dataset, test_dataset = train_test.split_train_and_test(balanced_dataset, 4000)

    print("train_dataset.shape:",train_dataset.shape)
    print("test_dataset.shape:", test_dataset.shape)
    np.savetxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv", train_dataset, delimiter=',', fmt='%f')
    np.savetxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv", test_dataset, delimiter=',', fmt='%f')




