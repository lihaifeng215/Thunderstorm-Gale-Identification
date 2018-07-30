'''
Created on 2018/3/30
by 大风识别项目组
通过给定的雷达回波数据的存储目录与对应自动在存储目录，画出当天00,30时刻带大风的RGB雷达图像
'''
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image,ImageDraw,ImageFont

COLORS = ['0,236,236', '1,160,246', '1,0,246', '0,233,0', '0,200,0', '0,144,0', '255,255,0', '231,192,0',
          '255,144,2',
          '255,0,0', '166,0,0', '101,0,0', '255,0,255', '153,85,201', '255,255,255']

def drawboard(board, colors,time,saveDir=r'./myFigures'):
    '''
    画图
    :param board: array float(int) 雷达数据矩阵 700*900
    :param colors: list 颜色类表
    :param time: string 当前雷达数据的时间以及高度组成的字符串 like '20170500_2500'，作为图片名称
    :param saveDir: string  保存图片的目录
    :return:
    '''
    #im = Image.new("RGB", (900, 700))  # 创建图片
    rgbs = np.array(np.zeros([700,900,3]),np.uint8)
    rgbs[:,:,:] = 255
    for i in range(5,75,5):
        judge = (board>=i) & (board < i+5)
        color = colors[i//5-1]
        rgb = [int(i) for i in color.split(',')]
        rgbs[judge,:] = rgb
    r = Image.fromarray(rgbs[:,:,0])
    g = Image.fromarray(rgbs[:,:,1])
    b = Image.fromarray(rgbs[:,:,2])
    im = Image.merge('RGB',(r,g,b))
    if os.path.exists(saveDir) == False:
        os.mkdir(saveDir)
    savePath = saveDir+'/%s.bmp'%time
    im.save(savePath)
    return savePath

def getNearstAWSFile(time):
    #time = RadarFile.split('_')[1]
    min = int(time[-2:])
    hour = time[-4:-2]
    other = time[:-4]
    for i in range(4):
        addmin = min + i
        submin = min - i
        if addmin%5==0:
            return other+hour+str(addmin).zfill(2)
        elif submin%5==0:
            return other+hour+str(submin).zfill(2)
# awstime = getNearstAWSFile('2017052105018')
# print(awstime,'lixian')

def drawWind(AWSDIR,OriginImagePath,saveDir='./myFigures',size=[3,1],wind=15):
    time = OriginImagePath.split('/')[-1].split('_')[0]
    date = time[0:-4]
    name = OriginImagePath.split('/')[-1].split('.')[0]
    AWSDIRDIRPath = os.path.join(AWSDIR,date)
    print('\n开始生成带大风图片，请耐心等待......')
    AWSfilenames = os.listdir(AWSDIRDIRPath)
    index = 0
    awstime = getNearstAWSFile(time) #获取最近的自动站时间
    savePath = ''
    for file in AWSfilenames:
        if awstime in file:  #自动站五分钟，对于雷达数据六分钟
            index = index + 1
            with Image.open(OriginImagePath).convert('RGBA') as im:
                lx = Image.new(im.mode, im.size)
                d = ImageDraw.Draw(lx)
                path = os.path.join(AWSDIRDIRPath, file)
                print(path)
                awsData = np.loadtxt(path, np.float32)
                for slice in awsData:
                    x = slice[2]  # 行
                    y = slice[1]  # 列
                    v = slice[8]  # 极大风速
                    if v >= wind:
                        d.rectangle((y - size[0], x - size[0], y + size[0], x + size[0]), (0,0,0), 'black')  # 左上右下
                    elif v >= 10 and v < wind:
                        d.ellipse((y - size[0], x - size[0], y + size[0], x + size[0]), None, 'purple')
                    elif v>=5 and v<=10:
                        d.ellipse((y - size[1], x - size[1], y + size[1], x + size[1]), None, 'brown')
                out = Image.alpha_composite(im, lx)
                if os.path.exists(saveDir) == False:
                    os.mkdir(saveDir)
                savePath = saveDir+'/NEW%s.bmp' % (name)
                out.save(savePath)
    return savePath


