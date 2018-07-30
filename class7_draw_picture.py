import numpy as np
import  matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageDraw

def AWS_to_Radar_time(AWS_time):
    time_dict = {'00': 0, '05': 1, '06': 0, '10': 2, '12': 0, '15': 3, '18': 0, '20': -2, '24': 0, '25': -1,
                 '30': 0, '35': 1, '36': 0, '40': 2, '42': 0, "45": 3,
                 '48': 0, "50": -2, '54': 0, '55': -1}
    return AWS_time + time_dict[str(AWS_time)[-2:]]

class draw_picture:
    def __init__(self):
        pass

    def draw_original_pic(self, image, colors):
        '''
        :param image: Radar image
        :param colors: color list
        :return: the matrix of RGB_image
        '''
        original_pic = np.ones((700,900,3),dtype=np.uint8) * 255
        for dbz in range(5,75,5):
            dbz_range = (image >= dbz) & (image < dbz + 5)
            color = colors[dbz//5 - 1].split(',')
            color = [int(c) for c in color]
            original_pic[dbz_range,:] = color
        return original_pic

    def draw_threshold_pic(self, image, colors, threshold=45):
        '''
        :param image: Radar image
        :param colors: color list
        :return: the matrix of RGB_image
        '''
        dbz_range = (image >= threshold)
        color = colors[-1].split(',')
        color = [int(c) for c in color]
        threshold_pic = self.draw_original_pic(image, colors)
        threshold_pic[dbz_range, :] = color
        return threshold_pic

    def draw_segment_pic(self, image, colors, threshold=45):
        '''
        :param image: Radar image
        :param colors: color list
        :return: the matrix of RGB_image
        '''
        segment_area = self.get_segment_area(image, threshold)
        segment_area = (segment_area == 1)
        color = colors[-1].split(',')
        color = [int(c) for c in color]
        segment_pic = self.draw_original_pic(image, colors)
        # print("segment_area == True:", segment_area[segment_area == True].shape)
        segment_pic[segment_area,:] = color
        return segment_pic

    # 画自动站风速与雷达对应图片
    def draw_AWS_wind_pic(self, image, Radar_time, colors, AWS_data):
        '''

        :param image: Radar image
        :param Radar_time:
        :param colors: color list
        :param AWS_data: the AWS matrix
        :return: the Image instance
        '''
        original_pic = self.draw_original_pic(image,colors)
        draw_AWS_wind = Image.fromarray(original_pic)
        AWS_wind_shape = ImageDraw.Draw(draw_AWS_wind)
        for record in AWS_data:
            AWS_time = int(record[0])
            x = int(record[2])
            y = int(record[3])
            wind = record[4]
            if AWS_to_Radar_time(AWS_time) == AWS_to_Radar_time(Radar_time):
                if wind >= 25:
                    AWS_wind_shape.rectangle((y - 5, x - 5, y + 5, x + 5), None, outline='black')
                elif (wind >= 15) & (wind < 25):
                    AWS_wind_shape.rectangle((y-3,x-3,y+3, x+3),None, outline='black')
                    print("wind:",wind)
                elif (wind >= 10) & (wind < 15):
                    AWS_wind_shape.ellipse((y-2,x-2,y+2,x+2),fill=(0,0,255), outline='purple')
                elif (wind >= 5) & (wind < 10):
                    AWS_wind_shape.ellipse((y-1,x-1,y+1,x+1), fill=None, outline='brown')
        return draw_AWS_wind

    # 画自动站上预测为大风的图片
    def draw_AWS_predict_wind_pic(self, image, time, colors, AWS_data):
        '''

        :param image: Radar image
        :param Radar_time:
        :param colors: color list
        :param AWS_data: the AWS matrix
        :return: the Image instance
        '''
        original_pic = self.draw_original_pic(image, colors)
        draw_AWS_predict_wind = Image.fromarray(original_pic)
        AWS_predict_wind_shape = ImageDraw.Draw(draw_AWS_predict_wind)
        for record in AWS_data:
            AWS_time = int(record[0])
            x = int(record[2])
            y = int(record[3])
            target = int(record[-2])
            predict = int(record[-1])
            if AWS_to_Radar_time(AWS_time) == AWS_to_Radar_time(time):
                if (predict == 1):
                    AWS_predict_wind_shape.rectangle((y-3,x-3,y+3,x+3), (0,0,0), outline='black')
                # if (predict == 0):
                #     AWS_predict_wind_shape.ellipse((y - 3, x - 3, y + 3, x + 3), (255,255,255), outline='blue')
                # if (predict == 1) and (target == 1):
                #     AWS_predict_wind_shape.rectangle((y-3,x-3,y+3,x+3), fill=(0,0,0), outline='black')
                # elif (predict == 1) and (target != 1):
                #     AWS_predict_wind_shape.ellipse((y-2,x-2,y+2,x+2), fill=None, outline='purple')
                # elif (predict != 1) and (target == 1):
                #     AWS_predict_wind_shape.ellipse((y-3,x-3,y+3,x+3), fill=(255,255,255), outline='red')
        return draw_AWS_predict_wind

    # # 画雷达图像上预测的大风图片
    # def draw_Radar_predict_wind_pic(self,):

    # 得到分割的飑线和风暴单体区域
    def get_segment_area(self, image, threshold=45):
        '''
        :param image: Radar_image
        :param threshold:
        :return:
        '''
        kernel_erosion_3 = np.ones([3, 3], dtype=np.uint8)
        kernel_erosion_5 = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],dtype=np.uint8)
        kernel_dilation_5 = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],dtype=np.uint8)
        kernel_dilation_7 = np.array([[0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],[0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
        img = np.copy(image)
        img[img < threshold] = 0
        img[img >= threshold] = 1
        erosion1 = cv2.erode(img, kernel=kernel_erosion_3, iterations=1)
        dilation1 = cv2.dilate(erosion1, kernel=kernel_dilation_7, iterations=6)
        dilation2 = cv2.dilate(dilation1, kernel=kernel_dilation_5, iterations=3)
        erosion2 = cv2.erode(dilation2, kernel=kernel_erosion_5, iterations=6)
        # print("erosion2 == 1", erosion2[erosion2 == 1].shape)
        pic1 = erosion2 * 255
        # a = Image.fromarray(pic1)
        # a.show()
        # segment_area = (erosion2 == 1) #变为 True False
        segment_area = erosion2
        return segment_area

    # 得到组合反射率
    def get_composite_r(self, path, time, hight_list):
        '''
        :param path: 存放多个ref数据的文件夹
        :param time: 时间（int）：例如：201705081806
        :param hight_list: 高度层
        :return: the matrix of RGB_image
        '''
        data_set = []
        for hight in hight_list:
            file_name = 'cappi_ref_' + str(AWS_to_Radar_time(time)) + '_' + str(hight) + '_0.ref'
            data = np.fromfile(os.path.join(path, file_name), dtype=np.uint8).reshape(700, 900)
            data[(data <= 0) | (data >= 80)] = 0
            data_set.append(data)
        print("___________%d___________" % time)
        data_set = np.array(data_set, dtype=np.uint8)
        print("data_set.shape:", data_set.shape)
        composite_r = np.max(data_set, axis=0)
        print("composite_r.shape:", composite_r.shape)
        print("np.max(composite_r):",np.max(composite_r))
        return composite_r

    # 得到特定高度层的图像
    def get_radar_layer(self, path, time, hight_list, layer):
        '''
        :param path: 存放多个ref数据的文件夹
        :param time: 时间（int）：例如：201705081806
        :param hight_list: 高度层
        :param layer: 0：1500,1:2500
        :return:
        '''
        data_set = []
        for hight in hight_list:
            file_name = 'cappi_ref_' + str(AWS_to_Radar_time(time)) + '_' + str(hight) + '_0.ref'
            data = np.fromfile(os.path.join(path, file_name), dtype=np.uint8).reshape(700, 900)
            data[(data <= 0) | (data >= 80)] = 0
            data_set.append(data)
        print("___________%d___________" % time)
        data_set = np.array(data_set, dtype=np.uint8)
        print("data_set.shape:", data_set.shape)
        return data_set[layer,:,:]



if __name__ == '__main__':
    colors = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0','255,144,2', '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '0,0,0']
    path = r'C:\Users\Helios\Desktop\项目\深圳气象局\2017.5.4全部雷达回波\20170504五分钟ref'
    AWS_wind_file = r'C:\Users\Helios\PycharmProjects\radarModel\Thunderstorm_winds_model4_5month\file\aws_data\get_all_aws_record_all.txt'
    # file_name = os.listdir(path)
    time_list = [201705040106, 201705040206, 201705040306, 201705040406,201705040506, 201705040606, 201705040706, 201705040806,201705040906, 201705041006, 201705041106, 201705041206]
    hight_list = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    for time in time_list:
        # data_set = []
        # for hight in hight_list:
        #     file_name = 'cappi_ref_'+ str(time) + '_' + str(hight) + '_0.ref'
        #     data = np.fromfile(os.path.join(path,file_name),dtype=np.uint8).reshape(700, 900)
        #     data[(data <= 0) | (data >= 80)] = 0
        #     data_set.append(data)
        # print("___________%d___________" %time)
        # data_set = np.array(data_set,dtype=np.uint8)
        # print("data_set.shape:", data_set.shape)
        # data_r = np.max(data_set, axis=0)
        # print("data_r.shape:", data_r.shape)
        # print(np.max(data_r))
        draw = draw_picture()
        data_r = draw.get_radar_layer(path,time,hight_list,1)

        # 画原始图片
        plt.figure(num=time, figsize=(20,15))
        plt.subplot(221)
        plt.title("origin_r")
        original_pic = draw.draw_original_pic(data_r, colors=colors)
        plt.imshow(original_pic)
        # 显示并保存原始图片
        ori_img = Image.fromarray(original_pic)
        # ori_img.show()
        # ori_img.save("file/original_image/%s.png" %time)

        # 画阈值分割图片
        plt.subplot(222)
        plt.title("threshold >= 45dbz")
        threshold_pic = draw.draw_threshold_pic(data_r, colors=colors, threshold=45)
        plt.imshow(threshold_pic)

        # 画飑线风暴图片
        plt.subplot(223)
        plt.title("segment the squall lines and stroms")
        setment_pic = draw.draw_segment_pic(data_r, colors=colors, threshold=45)
        plt.imshow(setment_pic)

        # 显示并保存画自动站大风图片
        plt.subplot(224)
        plt.title("AWS_wind_pic")
        AWS_data = np.loadtxt(AWS_wind_file, delimiter=',')
        AWS_wind_pic = draw.draw_AWS_wind_pic(data_r,colors=colors, Radar_time=time, AWS_data=AWS_data)
        plt.imshow(AWS_wind_pic)
        # AWS_wind_pic.save("file/AWS_wind_image/%s.png" %time)

        # 显示并保存四种对比图片
        # plt.show()
        plt.savefig("file/ori_thr_seg_wind_image/sample%s.png" %time)

        # 得到飑线风暴分割区域
        segment_area = draw.get_segment_area(data_r, threshold=45)
        # np.savetxt("file/segment_area/%s.csv" %time, segment_area, delimiter=',', fmt='%d')








