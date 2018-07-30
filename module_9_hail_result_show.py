import numpy as np
from matplotlib import pyplot as plt
from module.module_1_area_selection import Radar_features_append
from module.tools_draw_picture import draw_picture
from module.module_8_thunderstorm_result_show import build_show_sample
from PIL import Image, ImageDraw
import keras
import os
'''
1. 对图像中存在的冰雹区域进行识别
2. 输入：一张雷达图像
3. 输出：雷达图像中可能存在的冰雹区域。
'''
def draw_original_pic(image, colors):
    original_pic = np.ones((700, 900, 3), dtype=np.uint8) * 255
    for dbz in range(5, 75, 5):
        dbz_range = (image >= dbz) & (image < dbz + 5)
        color = colors[dbz // 5 - 1].split(',')
        color = [int(c) for c in color]
        original_pic[dbz_range, :] = color
    return original_pic

# 画自动站上预测为大风的图片
def draw_AWS_predict_wind_pic(image,colors, AWS_data):
    original_pic = draw_original_pic(image, colors)
    draw_AWS_predict_wind = Image.fromarray(original_pic)
    AWS_predict_wind_shape = ImageDraw.Draw(draw_AWS_predict_wind)
    for record in AWS_data:
        AWS_time = int(record[0])
        x = int(record[2])
        y = int(record[3])
        target = int(record[-2])
        predict = int(record[-1])
        if (predict == 1):
            AWS_predict_wind_shape.rectangle((y-3,x-3,y+3,x+3), fill=(0,0,0), outline='black')
        # elif (predict == 1) and (target != 1):
        #     AWS_predict_wind_shape.ellipse((y-2,x-2,y+2,x+2), fill=None, outline='purple')
        # elif (predict != 1) and (target == 1):
        #     AWS_predict_wind_shape.ellipse((y-3,x-3,y+3,x+3), fill=(255,255,255), outline='red')
    return draw_AWS_predict_wind

# 构造自动站数据
'''
get all AWS position
'''
def build_AWS_sample(Radar_image, Radar_time, radious):
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
def append_Radar_features(AWS_sample, radar_matrix, size):
    sub_image = []
    for i in AWS_sample:
        image = radar_matrix[i[2]-size:i[2] + size+1,i[3]-size:i[3] + size + 1]
        sub_image.append(image)
    sub_image = np.array(sub_image)
    print("sub_image.shape:", sub_image.shape)

    # 加上类标
    label = AWS_sample[:,4]
    data = np.column_stack((AWS_sample,sub_image.reshape(sub_image.shape[0],-1),label))
    return data


if __name__ == "__main__":

    Radar_time = 201705041006
    colors = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0',
              '255,144,2', '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '0,0,0']
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"F:\\2017radar"
    folder = str(Radar_time)[:-4] + 'ref'
    path = os.path.join(directory, folder)

    # 使用画图对象
    draw = draw_picture()
    # 得到组合反射率
    composite_r = draw.get_composite_r(path, Radar_time, hightList)

    show_sample = build_show_sample()
    AWS_sample = show_sample.build_AWS_sample(composite_r, Radar_time, 3)

    # 合并雷达特征
    # hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    # directory = r"F:\\2017radar"
    # size = 3
    # dataset = show_sample.append_Radar_features(AWS_sample, directory, hightList, size)
    #
    # # 机器学习模型
    # test_x = dataset[:, 5:-1]
    # test_y = dataset[:, -1]
    #
    # test_x = test_x.reshape(test_x.shape[0],9,7,7)
    # test_x = np.max(test_x,axis=1).reshape(test_x.shape[0],1,7,7).transpose(0,2,3,1)
    # print("test_x.shape:", test_x.shape)
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.externals import joblib
    # # test_x = MinMaxScaler().fit_transform(test_x)
    # model = keras.models.load_model("file1/models/model1_7x7.hdf5")
    # predict = model.predict(test_x)
    # result = np.column_stack((dataset, predict))
    # print("result.shape", result.shape)
    # result[1 - result[:, -1] < 1e-6, -1] = 1
    # print("hail number:", len(result[result[:, -1] == 1]))
    # result[1 - result[:, -1] < 1e-6, -1] = 1
    # draw_AWS_predict_wind = draw_AWS_predict_wind_pic(composite_r, colors, AWS_data=result)
    # draw_AWS_predict_wind.show()


    Radar_time = 201705081806
    data_r = np.loadtxt("file1/dataset_all_r.txt", dtype=int, delimiter=',')
    print("data_r.shape:", data_r.shape)
    data_r = data_r.reshape(data_r.shape[0], 700, 900)
    AWS_sample = build_AWS_sample(data_r[3], Radar_time, 3)
    print("AWS_sample.shape:", AWS_sample.shape)

    data = append_Radar_features(AWS_sample, data_r[3], 3)
    print("data.shape:", data.shape)
    np.savetxt("file1/test_data.csv", data, fmt='%d', delimiter=',')
    colors = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0','255,144,2', '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '0,0,0']
    # 画原始图片
    plt.figure()
    original_pic = draw_original_pic(data_r[3], colors=colors)
    plt.imshow(original_pic)
    # 显示并保存原始图片
    ori_img = Image.fromarray(original_pic)
    ori_img.show()
    # ori_img.save("file/original_image/%s.png" %time)
    features = data[:, 5:-1].reshape(data.shape[0], 1, 7, 7).transpose(0, 2, 3, 1)
    print("features.shape:",features.shape)
    label = data[:, -1]
    model = keras.models.load_model("file1/models/model1_7x7.hdf5")
    predict = model.predict(features)
    result = np.column_stack((data,predict))
    print("result.shape", result.shape)
    result[1 - result[:, -1] < 1e-6 , -1] = 1
    print("hail number:", len(result[result[:, -1] == 1]))
    result[1 - result[:, -1] < 1e-6, -1] = 1
    draw_AWS_predict_wind = draw_AWS_predict_wind_pic(data_r[3], colors, AWS_data=result)
    draw_AWS_predict_wind.show()
