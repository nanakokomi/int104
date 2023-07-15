import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#用于划分数据集
from sklearn.model_selection import train_test_split
#Kmeans 方法
from sklearn.cluster import KMeans
#评估指标：轮廓系数，前者为所有点的平均轮廓系数，后者返回每个点的轮廓系数
from sklearn.metrics import silhouette_score,silhouette_samples
#生成数据模块
from sklearn.datasets import make_blobs
#导入csv文件
df = pd.read_csv('./Data.csv', sep=',', header=0)
print(type(df))
print(type(df.values))


data = df.values #是csv文件里面的值
# 删除=2的
# mask=(data[:,-1]==2)
# newMa=np.delete(data,np.where(mask),axis=0)
# labels = newMa[:,-1]
# #删除index label
# newma = newMa[:,1:-1]
# data=newma[:,0:10]

#三种label放入三个list里面
class_0=[]
class_1=[]
# class_2=[]
#len 是矩阵的行数
for i in range(len(data)):
     #lable0
     if data[i,-1]==0:
         class_0.append(data[i,1:-1])#-1指的是倒数第一列
     elif data[i,-1]==1:
         class_1.append(data[i,1:-1])
#     elif data[i,-1]==2:
#          class_2.append(data[i,1:-1])
#可视化图
#plt.figure()
#plt.title('sum')
#plt.bar

#pca
plt.figure()
plt.title('dimensionality reduction')
pca =PCA(n_components=2)#实例化pca,变成2维
dataset = np.concatenate([class_0,class_1],axis=0)# 把0和1这个类通过这个函数作为一个大矩阵拼接起来
scaler = StandardScaler() #标准化
dataset = scaler.fit_transform(dataset)

newDataset = pca.fit_transform(dataset)
print(newDataset)
for i in range(len(class_0)):#class0
    plt.scatter(newDataset[i][0],newDataset[i][1],alpha=0.5,c='blue')#0是x轴，1是y轴，i是第几个数据
for i in range(len(class_0),len(class_0)+len(class_1),3):#class1 数据间隔是1
    plt.scatter(newDataset[i][0],newDataset[i][1],alpha=0.5,c='c')
plt.show()

#折线图
pca=PCA()
pca.fit(class_0,class_1)
plt.title('cumulative sum of explained ratio of principal component')
cumulative_explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_explained_var_ratio)
plt.show()

#分割数据
# x=data
# y=labels
# X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.4)

#分割数据法2
x=np.concatenate([class_0,class_1],axis=0)
train_x = pca.fit_transform(scaler.fit_transform(x))
labels_0_y = np.zeros(len(class_0),dtype=np.compat.long)
labels_1_y = np.zeros(len(class_1),dtype=np.compat.long)+1
y=np.concatenate([labels_0_y,labels_1_y],axis=0)
X_train, X_test, Y_train, Y_test = train_test_split(train_x,y, test_size=0.1)
#svm方法
model = SVC()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
r2 = recall_score(Y_test,y_pred)
f1i = f1_score(Y_test,y_pred)
print('accuracysvm:',accuracy)
print('r2',r2)
print("fi1:",f1i)

#desicion tree方法
clf = tree.DecisionTreeClassifier(criterion="entropy")
seel=clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
accuracy1 = clf.score(X_test,Y_test)
f12 = f1_score(Y_test,y_pred)
r22 = recall_score(Y_test,y_pred)
print("Accuracy1:",accuracy1)
print("fi2:",f12)
print('r2',r22)
#logistic
model = LogisticRegression()
model.fit(X_train,Y_train)
model.score(X_train,Y_train)
pre = model.predict(X_test)
accuracy2 = accuracy_score(Y_test,pre)
f13 = f1_score(Y_test,pre)
r23 = recall_score(Y_test,pre)
print("Accuracy2:",accuracy2)
print("fi3:",f13)
print('r2',r23)
# clf1 = GaussianNB
# put = clf1.fit(X_train, Y_train)
# y_pred = clf1.predict(X_test)
# accuracy2 = clf.score(X_test,Y_test)
# print("Accuracy2:",accuracy2)


#task 3

#用不同的n-clusters
# n_clusters = [x for x in range(3, 6)]
# for i in range(len(n_clusters)):
# # 实例化kmeans
#    classifier = KMeans(n_clusters=n_clusters[i])
#    classifier.fit(newDataset)
#    pred1 = classifier.predict(newDataset)
#
#   #绘制分类结果
#    plt.figure(figsize= (6, 6))
#    plt.scatter(newDataset[:, 0],newDataset[:, 1],c=pred1, s= 10)
#    plt.title("n_clusters= {}".format(n_clusters[i]))
#    plt.show()
#    # 打印平均轮廓系数
#    s = silhouette_score(newDataset, pred1)
#    print("When cluster= {}\nThe silhouette_score= {}".format(n_clusters[i], s))
# # 利用silhouette_samples计算轮廓系数为正的点的个数
#    n_s_bigger_than_zero = (silhouette_samples(newDataset, pred1) > 0).sum()
#    print("{}/{}\n".format(n_s_bigger_than_zero, newDataset.shape[0]))

