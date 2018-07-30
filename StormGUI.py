import tkinter
import numpy as np
from tkinter.filedialog import askdirectory,askopenfilename
from PIL import Image, ImageTk
import GUI.GUI.lxcanvas as lxc
from GUI.GUI.class3_append_label import append_label
from GUI.GUI.class7_draw_picture import draw_picture
from GUI.GUI.class8_build_show_sample import Radar_features_append
from GUI.GUI.class9_evaluation import Evaluation
from GUI.GUI.lxPredictDraw_erea import Add_draw
import os
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

RadarDIR = r'F:\2017radar\\' #所有的雷达数据目录
AWSDIR = r'F:\AWSdata十年数据' #所有自动站数据所在目录
Radar_time = 0
trainData = 0
originImagePath = '' #原始图像路径 D:/PyCharmIDE/PythonWorkSpace/RadarProgram/Data/figure2/201708101406_2500.bmp
preDictImage = 0
realWindImagePath = ''#真实大风图片路径

def Radar_to_AWS_time(AWS_time):
    time_dict = {'00': 0, '06': -1, '12':-2, '18': -3, '24': 1, '30': 0, '36': -1, '42': -2, '48': -3, '54': 1}
    return AWS_time + time_dict[str(AWS_time)[-2:]]

'''
input: the parent directory of each file
output: the records satisfied the standard
'''

# 选择和雷达时间对应的自动站记录
# def get_AWS_record(AWS_path,Radar_time, wind_speed):
#     files = os.listdir(AWS_path)
#     AWS_record = []
#     for name in files:
#         time = name.split('_')[1]
#         if int(time) == int(Radar_to_AWS_time(Radar_time)):
#             # attention : change the X and Y to rows and cols!!!!!!!!
#             data = np.loadtxt(os.path.join(AWS_path, name), dtype=float, delimiter=' ', usecols=[0, 2, 1, 8])
#             data = data[data[:, 3] >= wind_speed]
#             AWS_time = [int(time)] * data.shape[0]
#             data = np.column_stack((AWS_time, data))
#             # AWS_record.extend(data)
#             AWS_record = data
#     AWS_record = np.array(AWS_record)
#     print("the AWS_record.shape:", AWS_record.shape)
#     return AWS_record

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
    print("position.shape:", position.shape)

    # 只是用有回波的position
    position_area = []
    area = (Radar_image > 0) & (Radar_image < 80)
    print("area.shape:",area.shape)
    for i in range(position.shape[0]):
        if area[int(position[i, 0]), int(position[i, 1])] == True:
            position_area.append(position[i, :])

    position = np.array(position_area, dtype=int)
    time = np.zeros(position.shape[0], dtype=int) + Radar_time
    id = np.zeros((position.shape[0],), dtype=int)
    wind = np.zeros((position.shape[0],), dtype=int)
    AWS_sample = np.column_stack((time, id, position, wind))
    return AWS_sample


# 生成测试样本
def generateDataExample():
    global trainData, Radar_time, AWSDIR, RadarDIR
    # 得到自动站数据
    # AWS_folder = str(Radar_time)[:8]
    # AWS_path = os.path.join(AWSDIR, AWS_folder)
    # wind_speed = 0
    # AWS_sample = get_AWS_record(AWS_path,Radar_time,wind_speed)
    draw = draw_picture()
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    radious = 3
    path = os.path.join(RadarDIR,str(int(Radar_time))[:8]+'ref')
    Radar_image = draw.get_composite_r(path, Radar_time, hightList)
    AWS_sample = build_AWS_sample(Radar_image,Radar_time,radious=radious)


    # 将自动站合并特征
    directory = RadarDIR
    size = 3
    Radar = Radar_features_append()

    # 合并雷达数据
    subImage_all_dbz = Radar.find_subImage_all(AWS_sample, hightList, directory, size)
    data = Radar.append_R(AWS_sample, subImage_all_dbz)
    print("data.shape:", data.shape)

    # 去除回波不满足条件的数据
    data_denoised = Radar.remove_noise(data)
    print("data_denoised.shape:", data_denoised.shape)

    # 使用去噪后的数据添加特征
    subImage_all_dbz = data_denoised[:, 5:].reshape(data_denoised.shape[0], len(hightList), size * 2 + 1, size * 2 + 1)
    print("subImage_all_dbz.shape:", subImage_all_dbz.shape)
    subImage_all_b6m_dbz = Radar.find_subImage_all_b6m(data_denoised, hightList, directory, size)

    subImage_all_Z = Radar.dbz_to_Z(subImage_all_dbz)
    subImage_all_b6m_Z = Radar.dbz_to_Z(subImage_all_b6m_dbz)

    # get_R, delta_R, R_max, delta_R_max, R_max_height, delta_R_max_hight, get_Q, delta_Q
    # dbz
    data = Radar.append_delta_R(data_denoised, subImage_all_dbz, subImage_all_b6m_dbz)
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
    cls_label = append_label()
    append_reg_label = cls_label.append_regression_label(data)
    print("append_reg_label.shape:", append_reg_label.shape)
    statestr.set('样本生成成功！')
    trainData = append_reg_label
    return trainData

# 进行预测
def predict():
    global trainData, preDictImage,Radar_time,RadarDIR, originImagePath
    # img_open = Image.open(
    #     r'C:\Users\Helios\PycharmProjects\radarModel\GUI\GUI\myFigures\predictNEW201705040506_2500.bmp')
    # img_png = ImageTk.PhotoImage(img_open)
    # label_img.config(image=img_png)
    # label_img.image = img_png
    # assert 1==2

    # 分类
    # dataset = trainData
    # test_x = dataset[:, 5:-1]
    # test_y = dataset[:, -1]
    # test_y = np.array(test_y, dtype=int)
    # print("positive number:", len(test_y[test_y == 1]))
    # print("positive number:", test_y[test_y == 1])
    #
    # # 对测试集归一化！！！！！！
    # test_x= MinMaxScaler().fit_transform(test_x)
    #
    # model = joblib.load(r"C:\Users\Helios\PycharmProjects\radarModel\Segmentation\file\classifacation_models\LR_model2.pkl")
    # predict = model.predict(test_x)
    # output = np.column_stack((dataset[:, :5], test_y, predict))

    # 回归
    dataset = trainData
    test_x = dataset[:, 5:-1]
    test_y = dataset[:, -1]

    # 归一化
    test_x = MinMaxScaler().fit_transform(test_x)

    print("positive number:", len(test_y[test_y >= 15]))

    model = joblib.load(r"C:\Users\Helios\PycharmProjects\radarModel\GUI\GUI\models\Lasso_model.pkl")

    predict = model.predict(test_x)

    output = np.column_stack((dataset[:, :5], test_y, predict))

    np.savetxt("reg_dataset_to_draw_pic.csv", output, delimiter=',', fmt='%f')

    # 将回归结果变为分类结果
    # y_true
    output[output[:, -2] < 15, -2] = 0
    output[output[:, -2] >= 15, -2] = 1

    # y_pred
    output[output[:, -1] < 19, -1] = 0
    output[output[:, -1] >= 19, -1] = 1


    colors = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0',
              '255,144,2', '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '0,0,0']
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = RadarDIR
    time = Radar_time
    folder = str(time)[:-4] + 'ref'
    path = os.path.join(directory, folder)

    # 使用画图对象
    draw = draw_picture()
    # 得到组合反射率
    composite_r = draw.get_composite_r(path, time, hightList)

    # 直接画图
    draw_AWS_predict_wind = draw.draw_AWS_predict_wind_pic(composite_r, time, colors, AWS_data=output)
    draw_AWS_predict_wind.show()

    # 得到分割区域
    segment_area = draw.get_segment_area(composite_r, threshold=45)

    # 将不在飑线区域的大风去除掉
    output_removed = []
    for i in range(output.shape[0]):
        if segment_area[int(output[i, 2]), int(output[i, 3])] == 1:
            output_removed.append(output[i, :])
    output_removed = np.array(output_removed)
    np.savetxt("reg_dataset_to_draw_pic_removed.csv", output_removed, delimiter=',', fmt='%f')


    evaluation = Evaluation(output_removed[:,-2], output_removed[:,-1])
    evaluation.classification_eval()

    draw_AWS_predict_wind = draw.draw_AWS_predict_wind_pic(composite_r, time, colors, AWS_data=output_removed)
    preDictImage = draw_AWS_predict_wind

    # statestr.set('预测图片展示如下：')
    # img_png = ImageTk.PhotoImage(preDictImage)
    # label_img.config(image=img_png)
    # label_img.image = img_png
    sp = Add_draw(r'reg_dataset_to_draw_pic.csv',
                  originImagePath, 19)
    statestr.set('预测图片展示如下：')
    img_open = Image.open(sp)
    img_png = ImageTk.PhotoImage(img_open)
    label_img.config(image=img_png)
    label_img.image = img_png


def getInputs():
    str = pathentry.get()
    print(str)
    return str

def generateReal():
    global realWindImagePath,realFlag

    statestr.set('正在生成，请稍后......')
    realWindImagePath = lxc.drawWind(AWSDIR,originImagePath)
    statestr.set('真实大风图片展示如下：')
    img_open = Image.open(realWindImagePath)
    img_png = ImageTk.PhotoImage(img_open)
    label_img.config(image=img_png)
    label_img.image = img_png

def popPredictImage(event):
    preDictImage.show()
    # img_open = Image.open(r'C:\Users\Helios\PycharmProjects\radarModel\GUI\GUI\myFigures\predictNEW201705040506_2500.bmp')
    # img_png = ImageTk.PhotoImage(img_open)
    # label_img.config(image=img_png)
    # label_img.image = img_png

def popRealImage(event):
    im = Image.open(realWindImagePath)
    im.show()
# def go(event):
#     print(algorithmBox.get())

def selectPath():
    global originImagePath,realFlag,preFlag,trainDataPath, Radar_time
    path_ = askopenfilename()
    path.set(path_)
    pathstr = getInputs()
    if pathstr=='':
        return
    filename = pathstr.split('/')[-1]
    components = filename.split('_')
    time = components[2]+'_'+ components[3]
    Radar_time = int(components[2])
    imageName.set(time+'.bmp') #显示图片名称

    image = np.fromfile(pathstr, dtype=np.uint8).reshape(700, 900)
    originImagePath = lxc.drawboard(image,lxc.COLORS,time)
    img_open = Image.open(originImagePath)
    img_png = ImageTk.PhotoImage(img_open)
    label_img.config(image=img_png)
    label_img.image = img_png #keep a reference

    statestr.set('原始图像生成成功！')

    #显示位置
    # algorithmBox.grid(row=2,column=2)
    # algorithmBox.bind("<<ComboboxSelected>>", go)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)

    #生成
    generateButton = tkinter.Button(root, text='样本生成', command=generateDataExample)
    generateButton.grid(row=1, column=3)

    # 预测
    predictButton = tkinter.Button(root, text='预测', command=predict)
    predictButton.grid(row=3, column=2)
    predictButton.bind('<Double-Button-1>', popPredictImage)
    # 显示真实大风图片
    realButton = tkinter.Button(root, text='真实', command=generateReal)
    realButton.grid(row=3, column=3)
    realButton.bind('<Double-Button-1>', popRealImage)

if __name__ == '__main__':
    root = tkinter.Tk()
    windowWidth = 1000               #获得当前窗口宽
    windowHeight = 700              #获得当前窗口高
    screenWidth,screenHeight = root.maxsize()     #获得屏幕宽和高
    geometryParam = '%dx%d+%d+%d'%(windowWidth, windowHeight, (screenWidth-windowWidth)/2, (screenHeight - windowHeight)/2)
    root.geometry(geometryParam)    #设置窗口大小及偏移坐标
    #root.wm_attributes('-topmost',1)#窗口置顶

    path = tkinter.StringVar()
    tkinter.Label(root,text='预测图像路径').grid(row=1,column=0)
    pathentry = tkinter.Entry(root,textvariable = path)
    pathentry.grid(row=1,column=1)
    tkinter.Button(root,text='路径选择',command=selectPath).grid(row=1,column=2)

    #显示图片名称
    imageName = tkinter.StringVar()
    imageNameLabel = tkinter.Label(root,textvariable=imageName)
    imageNameLabel.grid(row=2,column=0)
    #显示当前状态
    statestr = tkinter.StringVar()
    stateLabel = tkinter.Label(root,textvariable=statestr)
    stateLabel.grid(row=2,column=1)


    # # 算法选择
    # number = tkinter.StringVar(root)
    # number.set('算法选择')
    # algorithmBox = ttk.Combobox(root, width=12, textvariable=number)
    # algorithmBox['values'] = ['DBN', 'SVM', 'LR']
    # algorithmBox.current(0)  # 选择第一个
    # print(algorithmBox.get())

    label_img = tkinter.Label(root)
    label_img.grid(row=4,rowspan=3,columnspan=3)

    #进入消息循环
    root.update_idletasks()
    root.mainloop()