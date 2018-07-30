import numpy as np
from module.module_1_area_selection import Radar_features_append
from module.module_3_thunderstorm_dataset import append_regression_label
from module.module_5_thunderstorm_model import get_batch_13x13, preprocess
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from module.tools_draw_picture import draw_picture
from sklearn import metrics
import os
'''
1. 调用训练好的模型，对某张雷达图像进行识别。
2. 输入：待识别雷达图像的9个高度层，训练好的模型
3. 雷暴大风识别的结果展示
'''
class build_show_sample:
    def __init__(self):
        pass

    # 构造自动站数据
    '''
    get all AWS position
    '''
    def build_AWS_sample(self, Radar_image, Radar_time, radious):
        rows = Radar_image.shape[0] // (radious * 2 + 1)
        cols = Radar_image.shape[1] // (radious * 2 + 1)
        rows = [r * (radious * 2 + 1) + radious for r in range(2, rows - 1)]
        cols = [c * (radious * 2 + 1) + radious for c in range(2, cols - 1)]
        print("len(rows):", len(rows))
        print("len(cols):", len(cols))
        position = []
        for r in rows:
            for c in cols:
                position.append([r,c])
        position = np.array(position, dtype=int)
        print("position.shape:", position.shape) # 网格化后中心位置点集
        # 只是用有回波的position
        position_area = []
        area = (Radar_image > 0) & (Radar_image < 80)
        print("area.shape:",area.shape)
        for i in range(position.shape[0]):
            if area[int(position[i, 0]), int(position[i, 1])] == True:
                position_area.append(position[i, :])

        # 将有回波的点集构造成自动站记录的形式。
        position = np.array(position_area, dtype=int)
        time = np.zeros(position.shape[0], dtype=int) + Radar_time
        id = np.zeros((position.shape[0],), dtype=int)
        wind = np.zeros((position.shape[0],), dtype=int)
        AWS_sample = np.column_stack((time, id, position, wind))
        return AWS_sample

    '''
    合并雷达特征
    '''
    def append_Radar_features(self, AWS_sample, Radar_dir, hightList, size):
        Radar = Radar_features_append()

        subImage_all_dbz = Radar.find_subImage_all(AWS_sample, hightList, Radar_dir, size)

        # dbz
        data = Radar.append_R(AWS_sample, subImage_all_dbz)
        print("data.shape:", data.shape)

        # 加上类标
        append_reg_label = append_regression_label(data)
        print("append_reg_label.shape:", append_reg_label.shape)
        return append_reg_label


if __name__ == '__main__':

    colors = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0',
              '255,144,2', '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '0,0,0']
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"F:\\2017radar"
    Radar_time = 201705040906
    folder = str(Radar_time)[:-4] + 'ref'
    path = os.path.join(directory, folder)

    # 使用画图对象
    draw = draw_picture()
    # 得到组合反射率
    composite_r = draw.get_composite_r(path, Radar_time, hightList)

    show_sample = build_show_sample()
    AWS_sample = show_sample.build_AWS_sample(composite_r,Radar_time, 3)

    # 合并雷达特征
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"F:\\2017radar"
    size = 6
    dataset = show_sample.append_Radar_features(AWS_sample, directory, hightList, size)

    # 机器学习模型
    test_x = dataset[:, 5:-1]
    test_y = dataset[:, -1]
    test_x = MinMaxScaler().fit_transform(test_x)
    model = joblib.load("file/regression_models/GBR_model.pkl")

    # 深度学习模型
    # import keras
    # model = keras.models.load_model('file/models/cnn_13x13.hdf5')
    # test_x,test_y = get_batch_13x13(dataset)
    # test_x = preprocess(test_x, 6, 9)

    predict = model.predict(test_x)
    output = np.column_stack((dataset[:,:5],test_y, predict))
    np.savetxt("file/models/reg_dataset_to_draw_pic.csv", output, delimiter=',', fmt='%f')

    # 将回归结果变为分类结果
    # y_true
    output[output[:, -2] < 19, -2] = 0
    output[output[:,-2] >= 19,-2] = 1

    # y_pred
    output[output[:, -1] < 19, -1] = 0
    output[output[:, -1] >= 19, -1] = 1

    # 直接画图
    layer_show = draw.get_radar_layer(path, Radar_time, hightList,2)
    draw_AWS_predict_wind = draw.draw_AWS_predict_wind_pic(layer_show, Radar_time, colors, AWS_data=output)
    draw_AWS_predict_wind.show()

    # 得到真实大风区域
    AWS_file = np.loadtxt("file/wind_speed_faster_than_5_2017year.csv",delimiter=',')
    AWS_wind = draw.draw_AWS_wind_pic(layer_show,Radar_time,colors,AWS_file)
    AWS_wind.show()










