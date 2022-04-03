from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score
from data_deal import data
from sklearn.metrics import classification_report
if __name__ == "__main__":
    x,y = data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(x_train, y_train)
    clf_pred = clf.predict(x_test)
    rfc = RandomForestClassifier(random_state=0)
    rfc = rfc.fit(x_train, y_train)
    rfc_pred = rfc.predict(x_test)
    print("Single Tree precision_score:{}".format(precision_score(clf_pred, y_test, average="micro")))
    print("Random Forest precision_score:{}".format(precision_score(rfc_pred, y_test,average="micro")))

    #进行十折交叉验证的数据预处理

    #使用十折交叉验证获取，max_depth（子树的最大深度）的最优取值
    # d_scores = []
    # for i in range(1,6):
    #     model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth = i)
    #     scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    #     d_scores.append(scores.mean())
    # print('max_depth分别取1，2，3，4，5时获得的准确率:')
    # print(d_scores)
    # print('最优值为： ',max(d_scores))
    # print('最优 max_depth 值为： ',d_scores.index(max(d_scores))+1)
    #
    # # 使用十折交叉验证获取，n_estimators（子树个数）的最优取值
    # n_scores = []
    # for i in range(1, 21):
    #     model = RandomForestClassifier(n_estimators= i, criterion='entropy', max_depth= 3)
    #     scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    #     n_scores.append(scores.mean())
    # print('n_estimators分别取 1~20 时获得的准确率:')
    # print(n_scores)
    # print('最优值为： ', max(n_scores))
    # print('最优 n_estimators 值为： ', n_scores.index(max(n_scores))+1)
