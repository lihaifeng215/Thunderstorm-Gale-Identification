import numpy as np
from Segmentation.class2_Radar_features import Radar_features_append
from Segmentation.class3_append_label import append_label
from sklearn.externals import joblib
from Segmentation.class7_draw_picture import draw_picture
from sklearn import metrics
import os



class build_show_sample:
    def __init__(self):
        pass
    # '''
    # get all AWS position
    # '''
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


    '''
    build AWS_sample
    '''
    def build_AWS_sample(self, AWS_position, AWS_time):
        time = np.zeros(AWS_position.shape[0], dtype=int) + AWS_time
        id = np.zeros((AWS_position.shape[0],),dtype=int)
        wind = np.ones((AWS_position.shape[0],),dtype=int) * 15
        AWS_sample = np.column_stack((time, id, AWS_position, wind))
        return AWS_sample

    '''
    get AWS_sample
    '''
    def get_AWS_sample(self, AWS_file, AWS_time):
        AWS_sample = AWS_file[AWS_file[:,0] == AWS_time]
        print("AWS_sample.shape:", AWS_sample.shape)
        return AWS_sample

    '''
    合并雷达特征
    '''


if __name__ == '__main__':

    # # 构建自动站样本
    # AWS_time = 201705081805
    # AWS_position = np.loadtxt('file/AWS_sample/AWS_position.csv', delimiter=',', dtype=int)
    # show_sample = build_show_sample()
    # AWS_sample = show_sample.build_AWS_sample(AWS_position, AWS_time)

    # 得到已有自动站样本
    #!!!!!!!  AWS_time必须以05结束  来抽取有效自动站
    AWS_time = 201705081805
    AWS_file = np.loadtxt('file/AWS_sample/wind_speed_faster_than_5_2017year.csv', delimiter=',')
    show_sample = build_show_sample()
    AWS_sample = show_sample.get_AWS_sample(AWS_file, AWS_time)



    # 合并雷达特征
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"J:\2017radar\\"
    size = 3
    Radar = Radar_features_append()

    subImage_all_dbz = Radar.find_subImage_all(AWS_sample, hightList, directory, size)
    subImage_all_b6m_dbz = Radar.find_subImage_all_b6m(AWS_sample, hightList, directory, size)

    subImage_all_Z = Radar.dbz_to_Z(subImage_all_dbz)
    subImage_all_b6m_Z = Radar.dbz_to_Z(subImage_all_b6m_dbz)

    # get_R, delta_R, R_max, delta_R_max, R_max_height, delta_R_max_hight, get_Q, delta_Q

    # dbz
    data = Radar.append_R(AWS_sample, subImage_all_dbz)
    data = Radar.append_delta_R(data, subImage_all_dbz, subImage_all_b6m_dbz)
    data = Radar.append_R_max(data, subImage_all_dbz)
    data = Radar.append_delta_R_max(data, subImage_all_dbz, subImage_all_b6m_dbz)
    data = Radar.append_R_max_hight(data, subImage_all_dbz, hightList)
    data = Radar.append_delta_R_max_hight(data, subImage_all_dbz, subImage_all_b6m_dbz, hightList)
    data = Radar.append_Q(data, subImage_all_dbz)
    data = Radar.append_delta_Q(data, subImage_all_dbz, subImage_all_b6m_dbz)

    # Z
    data = Radar.append_R(data, subImage_all_Z)
    data = Radar.append_delta_R(data, subImage_all_Z, subImage_all_b6m_Z)
    data = Radar.append_R_max(data, subImage_all_Z)
    data = Radar.append_delta_R_max(data, subImage_all_Z, subImage_all_b6m_Z)
    data = Radar.append_R_max_hight(data, subImage_all_Z, hightList)
    data = Radar.append_delta_R_max_hight(data, subImage_all_Z, subImage_all_b6m_Z, hightList)
    data = Radar.append_Q(data, subImage_all_Z)
    data = Radar.append_delta_Q(data, subImage_all_Z, subImage_all_b6m_Z)
    print("data.shape:", data.shape)

    # 加上类标
    label_type = 1
    cls_label = append_label()
    append_cls_label = cls_label.append_classification_label_wind(data, wind_speed=15)
    print("append_cls_label.shape:", append_cls_label.shape)

    # 进行预测
    dataset = append_cls_label
    test_x = dataset[:, 5:-1]
    test_y = dataset[:, -1]

    test_y = np.array(test_y, dtype=int)
    print("positive number:", len(test_y[test_y == 1]))
    print("positive number:", test_y[test_y == 1])

    model = joblib.load("file/classifacation_models/NB_model2.pkl")

    predict = model.predict(test_x)

    output = np.column_stack((dataset[:,:5],test_y, predict))

    np.savetxt("file/show_sample/dataset_to_draw_pic.csv", output, delimiter=',', fmt='%f')

    # 对结果进行后处理
    # 将不在飑线区域的预测结果去除掉


    # 画图
    colors = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0','255,144,2', '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '0,0,0']
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"J:\2017radar\\"
    time = 201705081805
    folder = str(time)[:-4] + 'ref'
    path = os.path.join(directory,folder)

    # 使用画图对象
    draw = draw_picture()
    # 得到组合反射率
    composite_r = draw.get_composite_r(path,time,hightList)

    # 直接画图
    draw_AWS_predict_wind = draw.draw_AWS_predict_wind_pic(composite_r, time, colors, AWS_data=output)
    draw_AWS_predict_wind.show()

    # 得到分割区域
    segment_area = draw.get_segment_area(composite_r,threshold=45)

    # 将不在飑线区域的大风去除掉
    output_removed = []
    for i in range(output.shape[0]):
        if segment_area[int(output[i,2]), int(output[i,3])] == 1:
            output_removed.append(output[i,:])
    output_removed = np.array(output_removed)

    draw_AWS_predict_wind = draw.draw_AWS_predict_wind_pic(composite_r,time,colors,AWS_data=output_removed)
    draw_AWS_predict_wind.show()

    # 得到真实大风区域
    AWS_wind = draw.draw_AWS_wind_pic(composite_r,time,colors,AWS_sample)
    AWS_wind.show()










