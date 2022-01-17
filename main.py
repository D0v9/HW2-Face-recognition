# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:15:27 2022

@author: inkzs
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import os, glob, random, cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

###載入img 並將train test分出來
def load_img(folder="D:\\T\\IP\\HW2-Face recognition\\att_faces"):  
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for k in range(40):
        folder2 = os.path.join(folder, 's%d' % (k + 1))
        data = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
        sample = random.sample(range(10), 5)
        x_train.extend([data[i].ravel() for i in range(10) if i in sample])
        x_test.extend([data[i].ravel() for i in range(10) if i not in sample])
        y_test.extend([k] * 5)
        y_train.extend([k] * 5)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


###PCA進行降維
def PCA_method(x_train,x_test,n_components):
    pca=PCA(n_components=n_components)
    x_train=pca.fit(x_train).transform(x_train)
    x_test=pca.transform(x_test)
    return x_train, x_test
    
###使用SVM做預測
def SVM(x_train,y_train,x_test,y_test,dimension):
    print("dim為",dimension)
    clf = SVC(kernel='linear', class_weight='balanced')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #print(classification_report(y_test, y_pred))
    print(" SVM準確率為:",accuracy_score(y_test, y_pred))
    return y_pred

###劃出混淆矩陣
def confusion(y_test,y_pred,dimension):
    confusion=confusion_matrix(y_test,y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(confusion,cmap='hot', alpha=0.3)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(x=j, y=i, s=confusion[i,j], va='center', ha='center')
    plt.xlabel('predicted label')        
    plt.ylabel('true label')
    dim=dimension
    plt.show()
    

    
###LDA進行轉換    
def LDA(x_train,x_test,y_train):
    lda=LinearDiscriminantAnalysis()
    x_train=lda.fit_transform(x_train,y_train)
    x_test=lda.transform(x_test)
    return x_train, x_test
    
#######################################################主程式

x_train, y_train, x_test, y_test = load_img()
num_train, num_test = x_train.shape[0], x_test.shape[0]

    
x_train10, x_test10 = PCA_method(x_train,x_test,10)
x_train20, x_test20 = PCA_method(x_train,x_test,20)
x_train30, x_test30 = PCA_method(x_train,x_test,30)
x_train40, x_test40 = PCA_method(x_train,x_test,40)
x_train50, x_test50 = PCA_method(x_train,x_test,50)    

y_pred10=SVM(x_train10,y_train,x_test10,y_test,10)
y_pred20=SVM(x_train20,y_train,x_test20,y_test,20)
y_pred30=SVM(x_train30,y_train,x_test30,y_test,30)
y_pred40=SVM(x_train40,y_train,x_test40,y_test,40)
y_pred50=SVM(x_train50,y_train,x_test50,y_test,50)


confusion(y_test,y_pred10,10)
confusion(y_test,y_pred20,20)
confusion(y_test,y_pred30,30)
confusion(y_test,y_pred40,40)
confusion(y_test,y_pred50,50)

print("%%%%%%%%%%%%%%%%%PCA+LDA%%%%%%%%%%%%%%%%")

x_trainLDA10, x_testLDA10 = LDA(x_train10,x_test10,y_train)
x_trainLDA20, x_testLDA20 = LDA(x_train20,x_test20,y_train)
x_trainLDA30, x_testLDA30 = LDA(x_train30,x_test30,y_train)
x_trainLDA40, x_testLDA40 = LDA(x_train40,x_test40,y_train)
x_trainLDA50, x_testLDA50 = LDA(x_train50,x_test50,y_train)

y_predLDA10=SVM(x_trainLDA10,y_train,x_testLDA10,y_test,10)
y_predLDA20=SVM(x_trainLDA20,y_train,x_testLDA20,y_test,20)
y_predLDA30=SVM(x_trainLDA30,y_train,x_testLDA30,y_test,30)
y_predLDA40=SVM(x_trainLDA40,y_train,x_testLDA40,y_test,40)
y_predLDA50=SVM(x_trainLDA50,y_train,x_testLDA50,y_test,50)

confusion(y_test,y_predLDA10,10)
confusion(y_test,y_predLDA20,20)
confusion(y_test,y_predLDA30,30)
confusion(y_test,y_predLDA40,40)
confusion(y_test,y_predLDA50,50)
