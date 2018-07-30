import numpy as np
import time
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
class classification_model:
    def __init__(self):
        pass

    '''
    Bayesian Ridge Regression
    '''

    def bayesian_ridge_regressor(self, train_x, train_y):
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge()
        model.fit(train_x, train_y)
        return model

    '''
    Decision Tree Regressor
    '''

    def decision_tree_regressor(self, train_x, train_y):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(train_x, train_y)
        return model

    '''
    Linear Regression
    '''

    def linear_regression(self, train_x, train_y):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(train_x, train_y)
        return model

    '''
    Ridge Regression
    '''

    def ridge_regression(self, train_x, train_y):
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
        model.fit(train_x, train_y)
        return model

    '''
    Lasso Regression
    '''

    def lasso_regression(self, train_x, train_y):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1)
        model.fit(train_x, train_y)
        return model

    '''
    Support Vector Regressor
    '''

    def support_vector_regressor(self, train_x, train_y):
        from sklearn.svm import SVR
        model = SVR()
        model.fit(train_x, train_y)
        return model

    '''
    K Nearest Neighbor Regressor
    '''

    def k_nearest_neighbor_regressor(self, train_x, train_y):
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(train_x, train_y)
        return model

    '''
    Random Forest Regressor
    '''

    def random_forest_regressor(self, train_x, train_y):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10)
        model.fit(train_x, train_y)
        return model

    '''
    AdaBoost Regressor
    '''

    def adaboost_regressor(self, train_x, train_y):
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=50)
        model.fit(train_x, train_y)
        return model

    '''
    Gradient Boosting Regressor
    '''

    def gradient_boosting_regressor(self, train_x, train_y):
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(train_x, train_y)
        return model

    # Normalize dataset
    def Normalize_dataset(self, dataset):
        return MinMaxScaler().fit_transform(dataset)

    # save model
    def save_model(self,model, model_name):
        joblib.dump(model, "file/regression_models/" + str(model_name) + '_model.pkl')

    # save error
    def save_error(self, error, error_name):
        np.savetxt("file/wind_error_classification/reg_error_%s_model.csv" % error_name, error, delimiter=',', fmt='%f')

    # use all classification models
    def use_regressors(self, train_x, train_y, test_x, test_y):

        classifiers = {'DTR': self.decision_tree_regressor,
                       'LR': self.linear_regression,
                       'Ridge': self.ridge_regression,
                       'Lasso': self.lasso_regression,
                       'BR': self.bayesian_ridge_regressor,
                       'SVR': self.support_vector_regressor,
                       'KNNR': self.k_nearest_neighbor_regressor,
                       'RFR': self.random_forest_regressor,
                       'AR': self.adaboost_regressor,
                       'GBR': self.gradient_boosting_regressor
                       }
        test_classifiers = ['DTR', 'LR', 'Ridge', 'Lasso', 'BR', 'SVR', 'KNNR', 'RFR', 'AR', 'GBR']
        for classifier in test_classifiers:
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            print('training took %fs!' % (time.time() - start_time))
            # 保存模型
            self.save_model(model, classifier)
            # 使用保存的模型预测
            model = joblib.load("file/regression_models/" + str(classifier) + '_model.pkl')
            test_time = time.time()
            predict = model.predict(test_x)
            score = model.score(test_x, test_y)
            mae = metrics.mean_absolute_error(test_y, predict)
            rmse = np.sqrt(metrics.mean_squared_error(test_y, predict))
            print('testing took %fs!' % (time.time() - test_time))
            print('score: %.2f, MAE: %.2f, RMSE: %.2f' % (score, mae, rmse))
            error = np.column_stack((test_y, predict))
            # 保存误差
            self.save_error(error, classifier)
            print('use all time: %fs!' % (time.time() - start_time))


if __name__ == '__main__':

    train_dataset = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv", delimiter=',')[:,5:]
    print("train_dataset.shape:", train_dataset.shape)
    test_dataset = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_reg_labeled_sub_7x7.csv", delimiter=',')[:, 5:]
    print("test_dataset.shape:", test_dataset.shape)
    train_x = train_dataset[:, 49*18:-1]
    print("train_x.shape:",train_x.shape)
    train_y = train_dataset[:, -1]
    train_y = np.array(train_y,dtype=int)
    test_x = test_dataset[:, 49*18:-1]
    test_y = test_dataset[:, -1]
    test_y = np.array(test_y,dtype=int)
    print("测试集中的正例比例：",len(test_y[test_y >= 15]) / len(test_y))

    cls_model = classification_model()
    train_x = cls_model.Normalize_dataset(train_x)
    test_x = cls_model.Normalize_dataset(test_x)
    cls_model.use_regressors(train_x, train_y, test_x, test_y)


