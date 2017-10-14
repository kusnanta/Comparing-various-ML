# from subprocess import check_output
# print(check_output(["ls", "src/inputData"]).decode("utf8"))

# import sys
# sys.stdout=open("output/mushrooms.csv","w")

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv("inputData/mushrooms.csv")
print (data.head(6))

print(data.isnull().sum())

print(data['class'].unique())

print(data.shape)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

print(data.head())

print(data['stalk-color-above-ring'].unique())

print(data.groupby('class').size())


'''
# Create a figure instance
fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))

# Create an axes instance and the boxplot
bp1 = axes[0,0].boxplot(data['stalk-color-above-ring'],patch_artist=True)

bp2 = axes[0,1].boxplot(data['stalk-color-below-ring'],patch_artist=True)

bp3 = axes[1,0].boxplot(data['stalk-surface-below-ring'],patch_artist=True)

bp4 = axes[1,1].boxplot(data['stalk-surface-above-ring'],patch_artist=True)
'''
ax = sns.boxplot(x='class', y='stalk-color-above-ring', data=data)
ax = sns.stripplot(x="class", y='stalk-color-above-ring', data=data, jitter=True, edgecolor="gray").set_title('lalala')
plt.title("Class w.r.t stalkcolor above ring",fontsize=12)
# plt.show()

X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only
X.head()
y.head()

print(X.describe())

y.head()

print(data.corr())

# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
print(X)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


covariance=pca.get_covariance()
#covariance

explained_variance=pca.explained_variance_
print(explained_variance)


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(22), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.show()

N=data.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
# plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], c= label_color)
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
