'''
注意PIL画图，是按照宽高来的，并不是按照行列来的
通过自动站数据，在雷达回波图像上圈出大风点与非大风点
'''
from PIL import Image,ImageDraw,ImageFont
import os
import numpy as np
# FileName = 'figure/0130.bmp'
# OutName = 'figure/new0130.bmp'
SIZE = 3
AWSDataDIR = r'J:\AWSdata十年数据'
def Add_draw(resultPath,imagePath,windThred):
    print(imagePath)
    print(imagePath.split('/'))
    name = imagePath.split('/')[-1].split('.')[0]
    date = name[:-9]
    time = str(int(name[:-5])-1)
    AWSDataDIRDIR = AWSDataDIR + '/'+date
    awsfiles = os.listdir(AWSDataDIRDIR)
    theWindAWSfile = ''
    print(time,'\n',awsfiles)
    for awsfile in awsfiles:
        # print("aws_file:",awsfile)
        if time in awsfile:
            # print("time:",time)
            theWindAWSfile = awsfile
            break
    if theWindAWSfile=='':
        print('图片对应的自动站数据不存在！！！')
        assert 1==2


    #画预测图片，针对大风点
    with Image.open(imagePath).convert('RGBA') as im:
        lx = Image.new(im.mode, im.size)
        d = ImageDraw.Draw(lx)
        result = np.loadtxt(resultPath,delimiter=',',usecols=[6, 2,3],dtype=np.float32)
        n,m = result.shape

        makeRec = np.zeros(shape=[700,900],dtype=np.float32)

        c = result[:,0]>windThred
        bigWindPos = result[c,:].copy().astype(np.int32)

        for i in range(n):

            v = result[i,0]
            x = int(result[i,-2])
            y = int(result[i,-1])
            #print(v,x,y)

            if v >windThred:
                d.ellipse((y - SIZE, x - SIZE, y + SIZE, x + SIZE), None, 'black')
                #print(x,y)
                makeRec[x, y] = 1
            # else:
            #     d.ellipse((y - SIZE, x - SIZE, y + SIZE, x + SIZE), None, 'blue')
        print(makeRec[makeRec!=0],bigWindPos,np.shape(bigWindPos))


        out = Image.alpha_composite(im,lx)
        savePath = './myFigures/predict_%s.bmp'%name
        out.save(savePath)

    ### 画预测图片，区域图片####
    num = np.shape(bigWindPos)[0]
    allPointsSets = []
    while (np.sum(bigWindPos[:, 0] == 0) < num):
        for i in range(num):
            v = bigWindPos[i, 0]
            if v != 0:
                bigWindPos[i, 0] = 0
                pointsSet = []
                startPoint = ( bigWindPos[i, -1].copy(),bigWindPos[i, -2].copy()) #列，行
                pointsSet = getPointsSet(startPoint, makeRec, 0)
                allPointsSets.append(pointsSet)
                #print(pointsSet)

        break
    print(allPointsSets)
    with Image.open(imagePath).convert('RGBA') as im:
        lx = Image.new(im.mode, im.size)
        d = ImageDraw.Draw(lx)

        for i in range(len(allPointsSets)):
            if(len(allPointsSets[i])>2):
                #print(allPointsSets[i][0:-1])
                print(allPointsSets[i])
                # start = allPointsSets[i][0]
                # points = allPointsSets[i]
                # points.append(start)
                # print(points)
                d.polygon(allPointsSets[i],fill='black',outline=None)
        out = Image.alpha_composite(im, lx)
        savePath = './myFigures/predict_%sArea.bmp' % name
        out.save(savePath)

    ###画真实图片，只带大风的###
    with Image.open(savePath).convert('RGBA') as im:
        lx = Image.new(im.mode, im.size)
        d = ImageDraw.Draw(lx)
        awspath = os.path.join(AWSDataDIRDIR, theWindAWSfile)
        result = np.loadtxt(awspath, usecols=[8, 2, 1], dtype=np.float32)
        # print(result)
        # assert 1==2
        n, m = result.shape

        for i in range(n):

            v = result[i, 0]
            x = int(result[i, -2])
            y = int(result[i, -1])
            # print(v,x,y)

            if v >= 15:
                d.ellipse((y - SIZE, x - SIZE, y + SIZE, x + SIZE), None, 'red')

        out = Image.alpha_composite(im, lx)
        saveBigWindPath = './myFigures/predict_%sOnlyRealBigWind.bmp' % name
        out.save(saveBigWindPath)
        # return savePath
        return saveBigWindPath

def getPointsSet(startPoint,makeRec,startInd):
    pointsSet = []
    #print('开始',startPoint)
    feetCon = 7
    #moveXY = np.array([[-feet, -feet], [-feet, 0], [-feet, feet], [0, feet], [feet, feet], [feet, 0], [feet, -feet], [0, -feet]])

    while( len(pointsSet)==0 or len(pointsSet)==1 or (startPoint in pointsSet)==False):

        pointsSet.append(startPoint)
        #print(startPoint)
        col, row = startPoint
        for j in range(1):
            #print(j)
            feet = feetCon
            #print(feet)
            moveXY = np.array([[-feet, -feet], [-feet, 0], [-feet, feet], [0, feet], [feet, feet], [feet, 0], [feet, -feet],[0, -feet]])
            n = np.shape(moveXY)[0]
            #print(moveXY)
            flag = 0
            for i in range(n):
                if startInd>=8:
                    startInd = startInd%8
                if startInd<=-9:
                    startInd = startInd + 8
                #print(startPoint,moveXY[startInd,1],moveXY[startInd,0],(int(col)+moveXY[startInd,1],int(row)+moveXY[startInd,0]),(col+1,row+1))

                rowNew = row + moveXY[startInd,0]
                colNew = col + moveXY[startInd,1]
                if(rowNew>=700 or colNew >=900 or rowNew <0 or colNew < 0):
                    flag = flag + 1
                    break
                #print('加减后的：',(col,row))
                #print(makeRec[row,col])

                if(makeRec[rowNew,colNew]==1):

                    startPoint = (colNew,rowNew)
                    #makeRec[rowNew, colNew] = -1
                    #print('进入后的',startPoint)
                    startInd = startInd-2
                    #print('开始索引',startInd)
                    flag = flag + 1
                    break
                startInd = startInd + 1
            if flag==1:
              break
    # if (len(pointsSet)<=2):
    #     makeRec[makeRec==-1] = 1
    return pointsSet
# 05081806 05082006
if __name__ == '__main__':
    sp = Add_draw(r'reg_dataset_to_draw_pic.csv',r'myFigures/201705040506_3500.bmp',19)
