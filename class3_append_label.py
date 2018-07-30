import numpy as np

class append_label:
    def __init__(self):
        pass

    '''
    append label to each record for classification
    '''
    def append_classification_label(self, origin_data, label_type):
        label = np.zeros((origin_data.shape[0],)) + label_type
        print("label.shape:", label.shape)
        append_cls_label = np.column_stack((origin_data, label))
        print("append classification label append_clf_label.shape:", append_cls_label.shape)
        return append_cls_label

    '''
    append label to each record for classification
    '''
    def append_classification_label_wind(self, origin_data, wind_speed):
        label = np.zeros((origin_data.shape[0],))
        label[origin_data[:,4] >= wind_speed] = 1
        print("label.shape:", label.shape)
        append_cls_label = np.column_stack((origin_data, label))
        print("append classification label append_clf_label.shape:", append_cls_label.shape)
        return append_cls_label

    '''
    append label to each record for classification
    '''
    def append_regression_label(self, origin_data):
        append_reg_label = np.column_stack((origin_data, origin_data[:,4]))
        print("append_regression_label.shape:", append_reg_label.shape)
        print("max wind speed:", np.max(append_reg_label[:,4]))
        print("min wind speed:", np.min(append_reg_label[:,4]))
        print("median wind speed:", np.median(append_reg_label[:,4]))
        return append_reg_label

if __name__ == '__main__':

    # # 分类
    # # 给正例加分类类标
    # label_type = 1
    # positive_sample = np.loadtxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features.csv", delimiter=',')
    # cls_label = append_label()
    # append_cls_label = cls_label.append_classification_label(positive_sample,label_type)
    # print("append_cls_label.shape:",append_cls_label.shape)
    # np.savetxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features_cls_labeled.csv", append_cls_label, delimiter=',', fmt='%f')
    #
    # # 给负例加分类类标
    # label_type = 0
    # negative_sample = np.loadtxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features.csv",delimiter=',')
    # print("negative_sample loaded")
    # cls_label = append_label()
    # append_cls_label = cls_label.append_classification_label(negative_sample, label_type)
    # print("append_cls_label.shape:", append_cls_label.shape)
    # np.savetxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features_cls_labeled.csv",append_cls_label, delimiter=',', fmt='%f')

    # 回归
    # 给正例加分类类标
    positive_sample = np.loadtxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features.csv",
                                 delimiter=',')
    cls_label = append_label()
    append_cls_label = cls_label.append_regression_label(positive_sample)
    print("append_reg_label.shape:", append_cls_label.shape)
    np.savetxt("file/Radar_append/positive_sample_17y_Radar_denoised_13x13_append_features_reg_labeled.csv",
               append_cls_label, delimiter=',', fmt='%f')

    # 给负例加分类类标
    negative_sample = np.loadtxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features.csv",
                                 delimiter=',')
    print("negative_sample loaded")
    cls_label = append_label()
    append_cls_label = cls_label.append_regression_label(negative_sample)
    print("append_reg_label.shape:", append_cls_label.shape)
    np.savetxt("file/Radar_append/negative_sample_05_17y_Radar_denoised_13x13_append_features_reg_labeled.csv",
               append_cls_label, delimiter=',', fmt='%f')








