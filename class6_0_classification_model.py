import numpy as np
import time
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
class classification_model:
    def __init__(self):
        pass

    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(self, train_x, train_y):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.001)
        model.fit(train_x, train_y)
        return model

    # KNN Classifier
    def knn_classifier(self, train_x, train_y):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        return model

    # Logistic Regression Classifier
    def logistic_regression_classifier(self, train_x, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
        model.fit(train_x, train_y)
        return model

    # Random Forest Classifier
    def random_forest_classifier(self, train_x, train_y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=8)
        model.fit(train_x, train_y)
        return model

    # Decision Tree Classifier
    def decision_tree_classifier(self, train_x, train_y):
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
        model.fit(train_x, train_y)
        return model

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self, train_x, train_y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model

    # SVM Classifier
    def svm_classifier(self, train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        model.fit(train_x, train_y)
        return model

    # SVM Classifier using cross validation
    def svm_cross_validation(self, train_x, train_y):
        from sklearn.grid_search import GridSearchCV
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        model.fit(train_x, train_y)
        return model

    # Normalize dataset
    def Normalize_dataset(self, dataset):
        return MinMaxScaler().fit_transform(dataset)

    # save model
    def save_model(self,model, model_name):
        joblib.dump(model, "file/classifacation_models/" + str(model_name) + '_model2.pkl')

    # save error
    def save_error(self, error, error_name):
        np.savetxt("file/wind_error_classification/error_%s_model.csv" % error_name, error, delimiter=',', fmt='%f')

    # use all classification models
    def use_classifiers(self, train_x, train_y, test_x, test_y):
        classifiers = {'NB': self.naive_bayes_classifier,
                       'KNN': self.knn_classifier,
                       'LR': self.logistic_regression_classifier,
                       'RF': self.random_forest_classifier,
                       'DT': self.decision_tree_classifier,
                       'SVM': self.svm_classifier,
                       'SVMCV': self.svm_cross_validation,
                       'GBDT': self.gradient_boosting_classifier
                       }
        test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
        for classifier in test_classifiers:
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            print('training took %fs!' % (time.time() - start_time))
            test_time = time.time()
            predict = model.predict(test_x)
            accuracy = metrics.accuracy_score(test_y, predict)
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('testing took %fs!' % (time.time() - test_time))
            # 保存模型
            self.save_model(model, classifier)
            print( 'accuracy: %.2f%%, precision: %.2f%%, recall: %.2f%%' % (100 * accuracy, 100 * precision, 100 * recall))

            error = np.column_stack((test_y, predict))
            # 保存误差
            self.save_error(error, classifier)
            print('use all time: %fs!' % (time.time() - start_time))


if __name__ == '__main__':

    train_dataset = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_cls_labeled_sub_7x7.csv", delimiter=',')[:,5:]
    print("train_dataset.shape:", train_dataset.shape)
    test_dataset = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_cls_labeled_sub_7x7.csv", delimiter=',')[:, 5:]
    print("test_dataset.shape:", test_dataset.shape)
    train_x = train_dataset[:, :-1]
    print("train_x.shape:",train_x.shape)
    train_y = train_dataset[:, -1]
    train_y = np.array(train_y,dtype=int)
    test_x = test_dataset[:, :-1]
    test_y = test_dataset[:, -1]
    test_y = np.array(test_y,dtype=int)
    print("测试集中的正例比例：",len(test_y[test_y == 1]) / len(test_y))

    cls_model = classification_model()
    train_x = cls_model.Normalize_dataset(train_x)
    test_x = cls_model.Normalize_dataset(test_x)

    cls_model.use_classifiers(train_x, train_y, test_x, test_y)


