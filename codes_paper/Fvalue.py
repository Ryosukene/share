import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import scipy.cluster.hierarchy as sch
import scipy.stats
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
pathroot='$share/predict/pathname/'

num=91#number of folder

electrodes=[]
for i in range(60):
    electrodes.append(i)
pred=[]
for i in range(num):
    data=torch.load(pathroot+str(i)+'/predicted_classid.pt')
    data_numpy = data.to('cpu').detach().numpy().copy()
    data_list = data_numpy.tolist()
    for j in range(len(data)):
        pred.append(data_list[j])  
#print('predicts',pred)
label=[]
for i in range(num):
    label_array=torch.load(pathroot+str(i)+'/labels.pt')
    label_array_numpy=label_array.to('cpu').detach().numpy().copy()
    label_array_list=label_array_numpy.tolist()
    for j in range(len(label_array)):
        label.append(label_array_list[j])
#print('labels',label)
confusion_matrix(label, pred)
# Accuracy ==============
accuracy = accuracy_score(label, pred)
# macro ==============
recall = recall_score(label, pred, average=None)  # 各クラスごとにprecisionが求まる

recall_macro = recall_score(label, pred, average='macro')  # macro平均で算出したpresicion

print(np.sum(recall) / 3 == recall_macro)  # True 　各クラスのrecallの平均 == recall_macro
precision = precision_score(label, pred, average=None)  # 各クラスごとにprecisionが求まる

precision_macro = precision_score(label, pred, average='macro')  # macro平均で算出したpresicion

print(np.sum(precision) / 3 == precision_macro)  # True 　各クラスのpresicionの平均 == precision_macro
f1 = f1_score(label, pred, average = None)
print('f1',f1)
f1_macro = f1_score(label, pred, average='macro')

print(np.sum(f1) / 3 == f1_macro)  # True 　各クラスのf1の平均 == f1_macro
# micro ==============
recall_micro = recall_score(label, pred, average='micro')

precision_micro = precision_score(label, pred, average='micro')

f1_micro = f1_score(label, pred, average='micro')
print('f1_micro',f1_micro)
print(accuracy == recall_micro == precision_micro == f1_micro)  # True