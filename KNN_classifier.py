# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import pandas as pd
#
# if __name__ == '__main__':
#     mine_data = pd.read_csv("feature.csv")
#     mine_label = pd.read_csv("label.csv")
#     X = mine_data
#     y = mine_label.values.ravel()
#     # 使用sklearn的切分函数进行划分数据集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#     # 使用sklearn的分类器进行预测
#     knn_clf = KNeighborsClassifier(n_neighbors=5)
#     knn_clf.fit(X_train, y_train)
#     y_predict = knn_clf.predict(X_test)
#     print(accuracy_score(y_test, y_predict))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score
import pandas as pd
from data_deal import data
if __name__ == '__main__':
    X,y = data()
    # 使用sklearn的切分函数进行划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    stdScaler = StandardScaler().fit(X_train)  # 创建规则
    mine_train_std = stdScaler.transform(X_train)  # 将规则应用到训练上
    mine_test_std = stdScaler.transform(X_train)  # 将规则应用到测试集上
    # 使用sklearn的分类器进行预测
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_predict = knn_clf.predict(X_test)
    print("knn precision_score:{}".format(accuracy_score(y_test, y_predict)))
