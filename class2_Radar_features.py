import numpy as np
class Radar_features:
    def __init__(self):
        pass

    '''
    get sub images of all layers
    '''
    def get_subImage_all(self, path, time, row, col, hightList, size):
        # path
        subImage_all = []
        for hight in hightList:
            data = np.fromfile(path + "cappi_ref_" + str(time) + '_' + str(hight) + "_0" + '.ref',dtype=np.int8).reshape(700, 900)
            # 当前高度层回波
            subImage_now = data[row - size:row + size + 1, col - size:col + size + 1].reshape((size * 2 + 1) ** 2, )
            # 所有高度层回波
            subImage_all.extend(subImage_now)
        subImage_all = np.array(subImage_all).reshape(len(hightList), size * 2 + 1, size * 2 + 1)
        subImage_all[((subImage_all < 0) | (subImage_all > 80))] = 0
        return subImage_all

    """
    get reflectivity
    """
    def get_R(self, subImage_all):
        print("subImage_all.shape", subImage_all.shape)
        return np.array(subImage_all).reshape(subImage_all.shape[0], -1)

    """
    get reflectivity difference of now and b6m
    """
    def delta_R(self, subImage_all, subImage_all_before):
        delta_r = subImage_all - subImage_all_before
        print("delta_r.shape:", delta_r.shape)
        return delta_r.reshape(subImage_all.shape[0], -1)

    """
    get absolute reflectivity difference of now and b6m
    """
    def abs_delta_R(self, subImage_all, subImage_all_before):
        abs_delta_r = np.abs(subImage_all - subImage_all_before)
        print("delta_r.shape:", abs_delta_r.shape)
        return abs_delta_r.reshape(subImage_all.shape[0], -1)

    '''
    get maximum reflectivity
    '''
    def R_max(self, subImage_all):
        subImage_max = np.max(subImage_all, axis=1)
        print("subImage_max.shape:", subImage_max.shape)
        return subImage_max.reshape(subImage_all.shape[0], -1)

    """
    get maximum reflectivity difference of now and b6m
    """
    def delta_R_max(self, subImage_all, subImage_all_before):
        delta_r_max = self.R_max(subImage_all) - self.R_max(subImage_all_before)
        print("delta_r.shape:", delta_r_max.shape)
        return delta_r_max.reshape(subImage_all.shape[0], -1)

    """
    get absolute maximum reflectivity difference of now and b6m
    """
    def abs_delta_R_max(self, subImage_all, subImage_all_before):
        abs_delta_r_max = np.abs(self.R_max(subImage_all) - self.R_max(subImage_all_before))
        print("abs_delta_r.shape:", abs_delta_r_max.shape)
        return abs_delta_r_max.reshape(subImage_all.shape[0], -1)

    '''
    get vertically integrated liquid
    '''
    def get_Q(self, subImage_all):
        q_max = (1000 * (3.44 * 10 ** -6)) * (np.sum((subImage_all ** (4 / 7)), axis=1))
        # q_max = (1000 * (3.44 * 10 ** -6)) * (np.sum(subImage_all,axis=1) - (subImage_all[0,:,:] + subImage_all[-1,:,:])/2)
        print("q_max.shape:", q_max.shape)
        return q_max.reshape(subImage_all.shape[0], -1)

    '''
    get delta Q_max
    '''
    def delta_Q(self, subImage_all, subImage_all_b6m):
        # 当前垂直液态水含量
        Q_now = self.get_Q(subImage_all)
        Q_b6m = self.get_Q(subImage_all_b6m)
        delta_q = Q_now - Q_b6m
        return delta_q.reshape(subImage_all.shape[0], -1)

    '''
    get absolute delta Q_max
    '''
    def abs_delta_Q(self, subImage_all, subImage_all_b6m):
        # 当前垂直液态水含量
        Q_now = self.get_Q(subImage_all)
        Q_b6m = self.get_Q(subImage_all_b6m)
        abs_delta_q = np.abs(Q_now - Q_b6m)
        return abs_delta_q.reshape(subImage_all.shape[0], -1)

    '''
    get height of maximum reflectivity
    '''
    def R_max_height(self, subImage_all, hightList):
        hightList = hightList[::-1]
        subImage_max_ind = np.argmax(subImage_all, axis=1)
        # subImage_height = subImage_max_ind * 1000 + hightList[0]
        # 最大回波的高度
        subImage_height = hightList[0] - subImage_max_ind * 1000
        print("hightList[0]:", hightList[0])
        print("subImage_height.shape:", subImage_height.shape)
        return subImage_height.reshape(subImage_all.shape[0], -1)

    '''
    get delta_R_hight
    '''
    def delta_R_max_hight(self, subImage_all, subImage_all_b6m, hightList):
        Q_now = self.R_max_height(subImage_all, hightList)
        Q_b6m = self.R_max_height(subImage_all_b6m, hightList)
        delta_r_hight = Q_now - Q_b6m
        return delta_r_hight.reshape(subImage_all.shape[0], -1)

    '''
    get absolute delta_R_hight
    '''
    def abs_delta_R_max_hight(self, subImage_all, subImage_all_b6m, hightList):
        Q_now = self.R_max_height(subImage_all, hightList)
        Q_b6m = self.R_max_height(subImage_all_b6m, hightList)
        abs_delta_r_hight = Q_now - Q_b6m
        return abs_delta_r_hight.reshape(subImage_all.shape[0], -1)

    '''
        dbz -> Z
        '''
    def dbz_to_Z(self, radar_dbz):
        return 10 ** (radar_dbz / 10)

    '''
    get time before 6 minutes
    '''
    def get_time_b6m(self, time):
        import datetime
        now = datetime.datetime.strptime(str(time), '%Y%m%d%H%M')
        b6m = now - datetime.timedelta(minutes=6)
        time_b6m = b6m.strftime('%Y%m%d%H%M')
        return time_b6m

    '''
    get time before 12 minutes
    '''
    def get_time_b12m(self, time):
        import datetime
        now = datetime.datetime.strptime(str(time), '%Y%m%d%H%M')
        b12m = now - datetime.timedelta(minutes=12)
        time_b12m = b12m.strftime('%Y%m%d%H%M')
        return time_b12m

    '''
    将自动站时间转换为对应的雷达时间
    input : int AWS_time
    output : int Radar_time
    '''
    def AWS_to_Radar_time(self, AWS_time):
        time_dict = {'00': 0, '05': 1, '06': 0, '10': 2, '12':0,  '15': 3,'18':0, '20': -2,'24':0, '25': -1, '30': 0, '35': 1,'36':0, '40': 2,'42':0, "45": 3,
                     '48':0,"50": -2,'54':0,'55': -1}
        return AWS_time + time_dict[str(AWS_time)[-2:]]

class Radar_features_append(Radar_features):
    def __init__(self):
        pass

    '''
    find_subImage_all
    '''
    def find_subImage_all(self, AWS_sample, hightList, directory, size):
        print("AWS_sample.shape:", AWS_sample.shape)
        dataset = []
        for i in range(AWS_sample.shape[0]):
            print("i:", i)
            # path = directory + str(int(AWS_sample[i][0]))[:8] + '\\'
            # 使用后缀有ref的雷达数据
            path = directory + str(int(AWS_sample[i][0]))[:8] + 'ref\\'
            print("path:",path)
            radar_sample1 = self.get_subImage_all(path, self.AWS_to_Radar_time(int(AWS_sample[i][0])), int(AWS_sample[i][2]), int(AWS_sample[i][3]), hightList, size)
            print(self.AWS_to_Radar_time(int(AWS_sample[i][0])),int(AWS_sample[i][2]),int(AWS_sample[i][3]))
            dataset.append(radar_sample1)
        sub_Image_all = np.array(dataset)
        print("find_subImage_all.shape:", sub_Image_all.shape)
        return sub_Image_all

    '''
    find_subImage_all_b6m
    '''
    def find_subImage_all_b6m(self, AWS_sample, hightList, directory, size):
        print("AWS_sample.shape:", AWS_sample.shape)
        dataset = []
        for i in range(AWS_sample.shape[0]):
            print(i)
            # path = directory + str(int(AWS_sample[i][0]))[:8] + '\\'
            # 使用后缀有ref的雷达数据
            path = directory + str(self.get_time_b6m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))))[:8] + 'ref\\'
            print("path:",path)
            radar_sample1 = self.get_subImage_all(path, self.get_time_b6m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))), int(AWS_sample[i][2]), int(AWS_sample[i][3]), hightList, size)
            print(self.get_time_b6m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))),int(AWS_sample[i][2]),int(AWS_sample[i][3]))
            dataset.append(radar_sample1)
        sub_Image_all_b6m = np.array(dataset)
        print("find_subImage_all_b6m.shape:", sub_Image_all_b6m.shape)
        return sub_Image_all_b6m

    '''
    find_subImage_all_b6m
    '''
    def find_subImage_all_b12m(self, AWS_sample, hightList, directory, size):
        print("AWS_sample.shape:", AWS_sample.shape)
        dataset = []
        for i in range(AWS_sample.shape[0]):
            print(i)
            # path = directory + str(int(AWS_sample[i][0]))[:8] + '\\'
            # 使用后缀有ref的雷达数据
            path = directory + str(self.get_time_b12m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))))[:8] + 'ref\\'
            print("path:",path)
            radar_sample1 = self.get_subImage_all(path, self.get_time_b12m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))), int(AWS_sample[i][2]), int(AWS_sample[i][3]), hightList, size)
            print(self.get_time_b12m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))),int(AWS_sample[i][2]),int(AWS_sample[i][3]))
            dataset.append(radar_sample1)
        sub_Image_all_b12m = np.array(dataset)
        print("find_subImage_all_b12m.shape:", sub_Image_all_b12m.shape)
        return sub_Image_all_b12m

    '''
    append R
    '''
    def append_R(self, origin_data, subImage_all):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.get_R(subImage_all)
        append_r = np.column_stack((origin_data, dataset))
        print("append_r.shape:", append_r.shape)
        return append_r

    '''
    append delta_R
    '''
    def append_delta_R(self, origin_data, subImage_all, subImage_all_before):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.delta_R(subImage_all, subImage_all_before)
        append_delta_r = np.column_stack((origin_data, dataset))
        print("append_delta_r.shape:", append_delta_r.shape)
        return append_delta_r

    '''
    append R_max
    '''
    def append_R_max(self, origin_data, subImage_all):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.R_max(subImage_all)
        append_r_max = np.column_stack((origin_data, dataset))
        print("append_r_max.shape:", append_r_max.shape)
        return append_r_max

    '''
    append delta_R_max
    '''
    def append_delta_R_max(self, origin_data, subImage_all, subImage_all_before):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.delta_R_max(subImage_all, subImage_all_before)
        append_delta_r_max = np.column_stack((origin_data, dataset))
        print("append_delta_r_max.shape:", append_delta_r_max.shape)
        return append_delta_r_max

    '''
    append R_max_hight
    '''
    def append_R_max_hight(self, origin_data, subImage_all, hightList):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.R_max_height(subImage_all,hightList)
        append_r_max_hight = np.column_stack((origin_data, dataset))
        print("append_r_max_hight.shape:", append_r_max_hight.shape)
        return append_r_max_hight

    '''
    append delta_R_max_hight
    '''
    def append_delta_R_max_hight(self, origin_data, subImage_all, subImage_all_before, hightList):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.delta_R_max_hight(subImage_all, subImage_all_before, hightList)
        append_delta_r_max_hight = np.column_stack((origin_data, dataset))
        print("append_delta_r_max_hight.shape:", append_delta_r_max_hight.shape)
        return append_delta_r_max_hight

    '''
    append Q
    '''
    def append_Q(self, origin_data, subImage_all):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.get_Q(subImage_all)
        append_q = np.column_stack((origin_data, dataset))
        print("append_q.shape:", append_q.shape)
        return append_q

    '''
    append delta_Q
    '''
    def append_delta_Q(self, origin_data, subImage_all, subImage_all_before):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.delta_Q(subImage_all, subImage_all_before)
        append_delta_q = np.column_stack((origin_data, dataset))
        print("append_delta_q.shape:", append_delta_q.shape)
        return append_delta_q

    '''
    remove the noise record
    '''
    def remove_noise(self, dataset):
        label = []
        for i in range(dataset.shape[0]):
            # 过滤条件
            if np.median(dataset[i, 5:]) < 1:
                label.append(i)
        # print(label)
        print("len(noise label):", len(label))
        label = np.array(label)
        print("noise label :", len(label) / dataset.shape[0])
        denoised_dataset = np.delete(dataset, label, axis=0)
        print("denoised_dataset.shape:", denoised_dataset.shape)
        return denoised_dataset
    '''
    合并所有需要的特征
    '''
    # def append_all_features(self, ):

if __name__ == '__main__':
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"J:\2017radar\\"
    size = 6
    # 待合并的自动站数据
    # AWS_sample = np.loadtxt("file/AWS_sample/negative_sample_05_17y_filtered.csv", delimiter=',')
    # print("AWS_sample.shape:", AWS_sample.shape)
    Radar = Radar_features_append()

    # # 合并雷达数据
    # subImage_all_dbz = Radar.find_subImage_all(AWS_sample, hightList, directory, size)
    # data = Radar.append_R(AWS_sample, subImage_all_dbz)
    # print("data.shape:", data.shape)
    #
    # # 去除回波不满足条件的数据
    # data_denoised = Radar.remove_noise(data)
    # print("data_denoised.shape:", data_denoised.shape)
    # np.savetxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13.csv", data_denoised, delimiter=',', fmt='%f')

    data_denoised = np.loadtxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13.csv", delimiter=',')
    print("data_denoised.shape:", data_denoised.shape)

    # 使用去噪后的数据添加特征
    subImage_all_dbz = data_denoised[:, 5:].reshape(data_denoised.shape[0], len(hightList), size * 2 + 1, size * 2 + 1)
    print("subImage_all_dbz.shape:",subImage_all_dbz.shape)
    subImage_all_b6m_dbz = Radar.find_subImage_all_b6m(data_denoised, hightList, directory, size)

    subImage_all_Z = Radar.dbz_to_Z(subImage_all_dbz)
    subImage_all_b6m_Z = Radar.dbz_to_Z(subImage_all_b6m_dbz)

    # get_R, delta_R, R_max, delta_R_max, R_max_height, delta_R_max_hight, get_Q, delta_Q
    # dbz
    # data = Radar.append_delta_R(data_denoised, subImage_all_dbz, subImage_all_b6m_dbz)
    # data = Radar.append_R_max(data, subImage_all_dbz)
    # data = Radar.append_delta_R_max(data, subImage_all_dbz, subImage_all_b6m_dbz)
    # data = Radar.append_R_max_hight(data, subImage_all_dbz, hightList)
    # data = Radar.append_delta_R_max_hight(data, subImage_all_dbz, subImage_all_b6m_dbz, hightList)
    # data = Radar.append_Q(data, subImage_all_dbz)
    # data = Radar.append_delta_Q(data, subImage_all_dbz, subImage_all_b6m_dbz)

    # Z
    data = Radar.append_R(data, subImage_all_Z)
    data = Radar.append_delta_R(data, subImage_all_Z, subImage_all_b6m_Z)
    data = Radar.append_R_max(data, subImage_all_Z)
    data = Radar.append_delta_R_max(data, subImage_all_Z, subImage_all_b6m_Z)
    # data = Radar.append_R_max_hight(data, subImage_all_Z, hightList)
    # data = Radar.append_delta_R_max_hight(data, subImage_all_Z, subImage_all_b6m_Z, hightList)
    data = Radar.append_Q(data, subImage_all_Z)
    data = Radar.append_delta_Q(data, subImage_all_Z, subImage_all_b6m_Z)

    print("data.shape:", data.shape)
    np.savetxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features.csv", data, delimiter=',', fmt='%f')







    # subImage_all_b6m = find_subImage_all_b6m(data, hightList, directory, size)
    # subImage_all = find_subImage_all(data, hightList, directory, size)
    # # get_R, delta_R, R_max, delta_R_max, R_max_height, delta_R_max_hight, get_Q, delta_Q
    # data = append_R(data, subImage_all)
    # data = append_delta_R(data, subImage_all, subImage_all_b6m)
    # data = append_R_max(data, subImage_all)
    # data = append_delta_R_max(data,subImage_all,subImage_all_b6m)
    # data = append_R_max_hight(data, subImage_all,hightList)
    # data = append_delta_R_max_hight(data, subImage_all, subImage_all_b6m, hightList)
    # data = append_Q(data, subImage_all)
    # data = append_delta_Q(data, subImage_all, subImage_all_b6m)
    # print("data.shape:", data.shape)
# np.savetxt("file/data_5m/wind_speed_dataset_5_denoise_append_features_all_3x3_no_features.csv", data, delimiter=',', fmt='%f')


