from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
from sklearn.metrics import confusion_matrix
import torch
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
pathroot='$share/predict/pathname/'
num=91#number of folder
predicts=[]
for i in range(num):
    data=torch.load(pathroot+str(i)+'/vector.pt')
    data_numpy = data.to('cpu').detach().numpy().copy()
    data_list = data_numpy.tolist()
    for j in range(len(data)):
        predicts.append(data_list[j]) 
        
_labels=[]
for i in range(num):
    label_array=torch.load(pathroot+str(i)+'/labels.pt')
    label_array_numpy=label_array.to('cpu').detach().numpy().copy()
    label_array_list=label_array_numpy.tolist()
    for j in range(len(label_array)):
        _labels.append(label_array_list[j])
        
colorname = 'viridis'
num = 7
data_new=np.array(predicts)           
data_new.shape
tsne = TSNE(n_components=2, random_state=1)
X_reduced = tsne.fit_transform(data_new)
df = pd.DataFrame(data={'x': X_reduced[:, 0],
                        'y': X_reduced[:, 1],
                        'label': _labels})
df=df.sort_values(by='label',ascending=True)
df['label']=df['label'].replace({0:'LP',1:'PM',2:'MS',3:'PC',4:'TC',5:'HV',6:'LV'})
plt.figure(figsize=(24,18))
sns.scatterplot(data=df, x='x', y='y', hue='label',
                palette=sns.color_palette('hls', 7))

plt.show()
