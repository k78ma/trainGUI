import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
import csv

from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
textname = 'embedding.csv'
path = 'data/'
txtinfile = path+textname
data = []
j = -1
with open(txtinfile,'r', encoding="utf-8") as fin:
    for row in csv.reader(fin, delimiter=','):
        if j ==-1:
            j = j+1
            continue
        else:
            data.append(row) # text descrip


X= np.array(data)
X = X.astype(np.float)
fig = plt.figure(1)
plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax = Axes3D(fig)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

textname = 'dataset.csv'
path = 'data/'
txtinfile = path+textname
data = []
j = -1
with open(txtinfile,'r',encoding="utf-8") as fin:
    for row in csv.reader(fin, delimiter=','):
        if j ==-1:
            j = j+1
            continue
        else:
            data.append(row[3:]) # text descrip
Y= np.array(data)
Y = Y.astype(np.float)

Y[Y != 0] = 1
# for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 1.5,
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(np.float)
col =['b','r']
for i in range(14):
    y = Y[:,i]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y*128, cmap=plt.cm.gist_rainbow,
            edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    name = str(i)+'.png'
    plt.show()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    plt.savefig(name)