import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from data_deal import data
#加载数据集
mine_data , mine_target = data()
#划分数据集
mine_data_train,mine_data_test,mine_target_train,mine_target_test = train_test_split(mine_data,mine_target)

#数据归一化处理
stdScaler = StandardScaler().fit(mine_data_train)  #创建规则
mine_train_std = stdScaler.transform(mine_data_train)   #将规则应用到训练上
mine_test_std = stdScaler.transform(mine_data_test)    #将规则应用到测试集上
#建立svm模型
svm = SVC().fit(mine_train_std,mine_target_train)
print('建立的模型是：',svm)
#预测训练集的结果
print('训练集的结果是：',svm.predict(mine_test_std))
print('训练集的真实值：',mine_target_test)
# print(svm.predict(cancer_test_std)==cancer_target_test)
#统计预测值和真实值一样的个数
num = np.sum(svm.predict(mine_test_std)==mine_target_test)
mine_target_predict = svm.predict(mine_test_std)
print('正确数为：',num)
print('错误数为：',(mine_test_std.shape[0]-num))
print('测试机的总样例数:',len(mine_test_std))
print('SVM precision_score：',precision_score(mine_target_test,mine_target_predict,average='micro'))