import numpy as np
from sklearn import metrics
'''
1. 雷暴大风识别使用回归相关的评测方法，冰雹识别使用分类相关评测方法
2. 输入：真实的结果和预测的结果
3. 输出：各类评估方法的指标
'''
# thunderstorm model evaluation
def MAE(y_true, y_pred):
    return "The MAE is %f." %metrics.mean_absolute_error(y_true, y_pred)
def RMSE(y_true, y_pred):
    return "The RMSE is %f." %np.sqrt(metrics.mean_squared_error(y_true, y_pred))
def R2_Score(y_true, y_pred):
    return "The R2_Score is %f." % metrics.r2_score(y_true, y_pred)

# hail model evaluation
def Accuracy(y_true, y_pred):
    return "The Accuracy is %f%%." % metrics.accuracy_score(y_true, y_pred) * 100
def Precision(y_true, y_pred):
    return "The Precision is %f%%." % metrics.precision_score(y_true, y_pred) * 100
def Recall(y_true, y_pred):
    return "The Recall is %f%%." % metrics.recall_score(y_true, y_pred) * 100
def F1_Score(y_true, y_pred):
    return "The F1_Score is %f%%." % metrics.f1_score(y_true, y_pred) * 100
