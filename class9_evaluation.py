import numpy as np
class Evaluation:
    def __init__(self, target, predict):
        self.target = target
        self.predict = predict
        assert target.shape == predict.shape, "test_y.shape != predict.shape!!!"
        self.xor = np.bitwise_xor(np.array(self.target, dtype=int), np.array(self.predict, dtype=int))
        self.bit_and = np.bitwise_and(np.array(self.target, dtype=int), np.array(self.predict, dtype=int))
        self.TP = self.bit_and[self.bit_and[:] == 1].shape[0]
        self.same = self.xor[self.xor[:] == 0].shape[0]
        self.differ = self.xor[self.xor[:] == 1].shape[0]

    '''
    Accuracy : 得到准确率
    '''
    def Accuracy(self):
        return self.same / (self.xor.shape[0] + 0.00001)

    '''
    Precision : 判断为正例的里面有多少是真的正例。
    '''
    def Precision(self):
        return self.TP / (self.predict[self.predict[:] == 1].shape[0] + 0.00001)

    '''
    Recall : 有多少比例的正例被判断为正例
    '''
    def Recall(self):
        return self.TP / (self.target[self.target[:] == 1].shape[0] + 0.00001)

    '''
    POD : 有多少比例的正例被判断为正例
    '''
    def POD(self):
        return self.Recall()

    '''
    FAR : 误识别率
    '''
    def FAR(self):
        return 1 - self.Precision()

    '''
    CSI : 临界成功指数
    '''
    def CSI(self):
        return self.TP / ((self.TP + self.differ) + 0.00001)

    '''
    Mean Abstract Error
    '''
    def MAE(self):
        return np.mean(np.abs(self.target - self.predict))

    '''
    Root of Mean Square Error
    '''
    def RMSE(self):
        return np.sqrt(np.mean(np.power(self.target - self.predict, 2)))

    '''
    classification_eval
    '''
    def classification_eval(self):
        print("Accuracy:%.2f%%, Precision:%.2f%%, Recall:%.2f%%, POD:%.2f%%, FAR:%.2f%%, CSI:%.2f%%" %(self.Accuracy() * 100, self.Precision() * 100, self.Recall() * 100, self.POD() * 100, self.FAR() * 100, self.CSI() * 100))

# test class
if __name__ == "__main__":
    error = np.loadtxt("file/wind_error_classification/error_GBDT_model.csv", delimiter=',')
    e = Evaluation(error[:,0], error[:,1])
    e.classification_eval()

