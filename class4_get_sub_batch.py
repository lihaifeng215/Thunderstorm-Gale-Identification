import numpy as np

class get_sub_batch():
    def __init__(self):
        pass

    # 得到 origin_size * 2 + 1 -> target_size * 2 + 1 的子区域
    def get_sub_batch(self, data, origin_size, target_size, channel):
        features = data[:, 5:-1]
        features = features.reshape(data.shape[0], channel, origin_size * 2 + 1, origin_size * 2 + 1)
        print("features.shape:", features.shape)
        sub_features = features[:, :, origin_size - target_size: origin_size + target_size + 1,
                       origin_size - target_size: origin_size + target_size + 1]
        print("sub_features.shape:", sub_features.shape)
        sub_features = sub_features.reshape(data.shape[0], -1)
        print("sub_features.shape:", sub_features.shape)
        sub_batch = np.column_stack((data[:, :5], sub_features, data[:, -1]))
        print("sub_batch.shape:", sub_batch.shape)
        return sub_batch
if __name__ == '__main__':

    # 得到正样本子区域
    sub_batch = get_sub_batch()
    data = np.loadtxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features_reg_labeled.csv",delimiter=',')
    print("data.shape:", data.shape)
    sub_batch = sub_batch.get_sub_batch(data, 6, 3, 48)
    np.savetxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv",sub_batch,delimiter=',',fmt='%f')

    # 得到负样本子区域
    sub_batch = get_sub_batch()
    data = np.loadtxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features_reg_labeled.csv",delimiter=',')
    print("data.shape:", data.shape)
    sub_batch = sub_batch.get_sub_batch(data, 6, 3, 48)
    np.savetxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv",sub_batch, delimiter=',', fmt='%f')

