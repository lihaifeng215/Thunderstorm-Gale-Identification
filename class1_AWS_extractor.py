import numpy as np
import os
class AWS_extractor:
    def __init__(self):
        pass

    '''
    将自动站时间转换为对应的雷达时间
    input : int AWS_time
    output : int Radar_time
    '''
    def AWS_to_Radar_time(self, AWS_time):
        time_dict = {'00': 0, '05': 1, '06': 0, '10': 2, '12': 0, '15': 3, '18': 0, '20': -2, '24': 0, '25': -1,
                     '30': 0, '35': 1, '36': 0, '40': 2, '42': 0, "45": 3,
                     '48': 0, "50": -2, '54': 0, '55': -1}
        return AWS_time + time_dict[str(AWS_time)[-2:]]

    '''
    input: the parent directory of each file
    output: the records satisfied the standard
    '''
    def get_AWS_record(self, AWS_path, wind_speed):
        files = os.listdir(AWS_path)
        AWS_record = []
        for name in files:
            time = name.split('_')[1]

            # 只选择所需时刻的自动站记录(去除条件后得到所有自动站记录)
            # if int(time[-2:]) != 5:
            #     continue
            # print(name)

            # attention : change the X and Y to rows and cols!!!!!!!!
            data = np.loadtxt(os.path.join(AWS_path, name), dtype=float, delimiter=' ', usecols=[0, 2, 1, 8])
            data = data[data[:, 3] >= wind_speed]
            AWS_time = [int(time)] * data.shape[0]
            data = np.column_stack((AWS_time, data))
            AWS_record.extend(data)
        AWS_record = np.array(AWS_record)
        print("the AWS_record.shape:", AWS_record.shape)
        return AWS_record

    '''
    get AWS data of wind speed more than  5 m/s
    '''
    def get_AWS_sample(self, path, wind_speed):
        files = os.listdir(path)
        print("files:",files)
        AWS_sample = []
        for f in files:

            # 只选择所需年份的自动站（去除条件后得到所有年份记录）
            time = f.split('-')[1]
            if int(time[:4]) != 2017:
                continue

            print("AWS_path:", os.path.join(path,f))
            AWS_record = self.get_AWS_record(os.path.join(path,f), wind_speed)
            AWS_sample.extend(AWS_record)
        AWS_sample = np.array(AWS_sample)
        return AWS_sample

    '''
    delete 12,1,2 month AWS_sample
    '''
    def delete_month(self, AWS_sample, month_list):
        delete = []
        for i in range(AWS_sample.shape[0]):
            if str(AWS_sample[i,0])[4:6] in month_list:
                # print("AWS_sample[i,0]:", str(AWS_sample[i,0]))
                delete.append(i)
        print("delete_month_number:", len(delete))
        filtered_month = np.delete(AWS_sample, delete, axis=0)
        return filtered_month

    '''
    delete typhoon day AWS_sample
    '''
    def delete_typhoon(self, AWS_sample, typhoon_list):
        delete = []
        for i in range(AWS_sample.shape[0]):
            if str(AWS_sample[i,0])[0:8] in typhoon_list:
                # print("AWS_sample[i,0]:", str(AWS_sample[i,0]))
                delete.append(i)
        print("delete_typhoon_number:", len(delete))
        filtered_typhoon = np.delete(AWS_sample, delete, axis=0)
        return filtered_typhoon

    '''
    delete not first time big wind record
    下一条和上一条记录风速一致，id一致则删去
    '''
    def delete_not_first_time(self, AWS_sample):
        sorted_AWS_sample = sorted(AWS_sample, key=lambda x : (x[1], x[0]))
        sorted_AWS_sample = np.array(sorted_AWS_sample)
        previous_record = sorted_AWS_sample[0,:]
        delete = []
        for i in range(sorted_AWS_sample.shape[0]):
            record = sorted_AWS_sample[i,:]
            if record[1] == previous_record[1] and record[4] == previous_record[4]:
               delete.append(i)
            previous_record = record
        print("delete_not_first_time_record number:", len(delete))
        filtered_not_first_time = np.delete(sorted_AWS_sample, delete, axis=0)
        return filtered_not_first_time

    '''
    得到所有时刻的正样本
    '''
    def get_positive_sample(self, path, wind_speed, month_list, typhoon_list):

        # 得到15m/s以上的风速记录
        AWS_sample = self.get_AWS_sample(path, wind_speed)
        print("AWS_sample.shape:", AWS_sample.shape)

        # 删除12月 1月 2月的风速记录
        filtered_month = self.delete_month(AWS_sample, month_list)
        print("filtered_month.shape:", filtered_month.shape)

        # 删除有台风的日期的风速记录
        filtered_typhoon = self.delete_typhoon(filtered_month, typhoon_list)
        print("filtered_typhoon.shape:", filtered_typhoon.shape)

        # 删除非首次出现的大风
        filtered_not_first_time = self.delete_not_first_time(filtered_typhoon)
        print("filtered_not_first_time.shape:", filtered_not_first_time.shape)

        # 按时间进行排序
        positive_sample = filtered_not_first_time[np.lexsort(filtered_not_first_time[:,::-1].T)]
        print("positive_sample.shape:", positive_sample.shape)

        return positive_sample

    '''
    得到5分钟时刻的负样本
    '''
    def get_negative_sample(self, path, wind_speed, month_list, typhoon_list):
        # 得到5m/s以上的风速记录
        AWS_sample = self.get_AWS_sample(path, wind_speed)
        print("AWS_sample.shape:", AWS_sample.shape)

        # 得到5 - 15m/s的风速记录
        small_wind_05 = AWS_sample[AWS_sample[:, 4] < 15]
        print("small_wind_05.shape:", small_wind_05.shape)

        # 删除12月 1月 2月的风速记录
        filtered_month = self.delete_month(small_wind_05, month_list)
        print("filtered_month.shape:", filtered_month.shape)

        # 删除有台风的日期的风速记录
        filtered_typhoon = self.delete_typhoon(filtered_month, typhoon_list)
        print("filtered_typhoon.shape:", filtered_typhoon.shape)

        return filtered_typhoon

    '''
    get all AWS position
    '''
    # def get_AWS_position(self, AWS_file):
    #     x = AWS_file[:, 2]
    #     y = AWS_file[:, 3]
    #     position = np.column_stack((x, y))
    #     position_tuple = [tuple(i) for i in position]
    #     AWS_position = set(position_tuple)
    #     AWS_position = [[p[0],p[1]] for p in AWS_position]
    #     AWS_position = np.array(AWS_position, dtype=int)
    #     print("AWS_position.shape:", AWS_position.shape)
    #     return AWS_position





# if __name__ == '__main__':
    # path = r"E:\自动站数据15-17"
    # wind_speed = 5
    # positive_wind_speed = 15
    # month_list = ['12', '01', '02']
    # typhoon_list = ['20170611', '20170612', '20170613', '20170722', '20170723', '20170724', '20170823', '20170824', '20170827', '20151004', '20150708', '20150709', '20150710', '20161021', '20161022', '20160801', '20160802']
    # AWS = AWS_extractor()

    # !!!!!! 使用之前需要去掉 get_AWS_record()方法 中的5min时刻注释
    # positive_sample = AWS.get_positive_sample(path,positive_wind_speed, month_list,typhoon_list)
    # print("positive_sample.shape:", positive_sample.shape)
    # np.savetxt("file/AWS_sample/positive_sample_17y_filtered.csv", positive_sample, delimiter=',',fmt='%f')

    # !!!!!! 使用之前需要加上 get_AWS_record()方法 中的5min时刻注释
    # negative_sample_05 = AWS.get_negative_sample(path, wind_speed, month_list, typhoon_list)
    # print("negative_sample_05.shape:", negative_sample_05.shape)
    # np.savetxt("file/AWS_sample/negative_sample_05_17y_filtered.csv", negative_sample_05, delimiter=',',fmt='%f')






    # # 得到所有的自动站位置
    # AWS_file = np.loadtxt("file/AWS_sample/negative_sample_05_17y_filtered.csv", delimiter=',')
    # AWS_position = AWS.get_AWS_position(AWS_file)
    # np.savetxt("file/AWS_sample/AWS_position.csv", AWS_position, delimiter=',',fmt='%d')


    # # 得到5m/s以上的风速记录
    # AWS_sample = AWS.get_AWS_sample(path, wind_speed)
    # print("AWS_sample.shape:", AWS_sample.shape)
    # np.savetxt("file/AWS_sample/wind_speed_faster_than_5_2017year.csv", AWS_sample, delimiter=',', fmt='%f')
    #
    # AWS_sample = np.loadtxt("file/AWS_sample/wind_speed_faster_than_5_2017year.csv", delimiter=',')
    # print("AWS_sample.shape:", AWS_sample.shape)
    #
    # # 得到15m/s以上的风速记录
    # big_wind_05 = AWS_sample[AWS_sample[:, 4] >= 15]
    # print("big_wind_05.shape:", big_wind_05.shape)
    # np.savetxt("file/AWS_sample/wind_speed_faster_than_5_2017year_big_wind.csv", big_wind_05, delimiter=',', fmt='%f')
    #
    # # 删除12月 1月 2月的风速记录
    # filtered_month = AWS.delete_month(AWS_sample, month_list)
    # print("filtered_month.shape:", filtered_month.shape)
    #
    # # 删除12月 1月 2月后的大风记录数
    # filtered_month_big_wind_05 = filtered_month[filtered_month[:, 4] >= 15]
    # print("filtered_month_big_wind_05.shape:", filtered_month_big_wind_05.shape)
    #
    # # 删除有台风的日期的风速记录
    # filtered_typhoon = AWS.delete_typhoon(filtered_month, typhoon_list)
    # print("filtered_typhoon.shape:", filtered_typhoon.shape)
    #
    # # 删除12月1月2月及有台风的日期后的大风记录数
    # filtered_typhoon_big_wind_05 = filtered_typhoon[filtered_typhoon[:, 4] >= 15]
    # print("filtered_typhoon_big_wind_05.shape:", filtered_typhoon_big_wind_05.shape)
    # np.savetxt("file/AWS_sample/filtered_month_typhoon_big_wind_3year_5min.txt",filtered_typhoon_big_wind_05, delimiter=',',fmt='%f')





    # path = r"E:\自动站数据15-17"
    # wind_speed = 15
    # month_list = ['12', '01', '02']
    # typhoon_list = ['20170611', '20170612', '20170613', '20170722', '20170723', '20170724', '20170823', '20170824',
    #                 '20170827', '20151004', '20150708', '20150709', '20150710', '20161021', '20161022', '20160801',
    #                 '20160802']
    # AWS = AWS_extractor(path, wind_speed, month_list, typhoon_list)
    # # 得到15m/s以上的风速记录
    # AWS_sample = AWS.get_AWS_sample(path, wind_speed)
    # print("AWS_sample.shape:", AWS_sample.shape)
    # np.savetxt("file/AWS_sample/wind_speed_faster_than_15_2017year.csv", AWS_sample, delimiter=',', fmt='%f')
    #
    # AWS_sample = np.loadtxt("file/AWS_sample/wind_speed_faster_than_15_2017year.csv", delimiter=',')
    # print("AWS_sample.shape:", AWS_sample.shape)
    #
    #
    # # 删除12月 1月 2月的风速记录
    # filtered_month = AWS.delete_month(AWS_sample, month_list)
    # print("filtered_month.shape:", filtered_month.shape)
    #
    # # 删除有台风的日期的风速记录
    # filtered_typhoon = AWS.delete_typhoon(filtered_month, typhoon_list)
    # print("filtered_typhoon.shape:", filtered_typhoon.shape)
    #
    # # 删除非首次出现的大风
    # filtered_not_first_time = AWS.delete_not_first_time(filtered_typhoon)
    # print("filtered_not_first_time.shape:",filtered_not_first_time.shape)
    # np.savetxt("file/AWS_sample/positive_sample_17y_filtered.csv", filtered_not_first_time, delimiter=',',fmt='%f')


