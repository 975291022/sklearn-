from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from data_deal import data
from sklearn.metrics import accuracy_score,precision_score
mine_data,mine_label = data()
X_train, X_test, y_train, y_test = train_test_split(mine_data,mine_label, test_size=0.2)
stdScaler = StandardScaler().fit(X_train)  #创建规则
mine_train_std = stdScaler.transform(X_train)   #将规则应用到训练上
mine_test_std = stdScaler.transform(X_train)    #将规则应用到测试集上
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("NaiveBayes: %lf" % (precision_score(y_pred, y_test,average="micro")))
