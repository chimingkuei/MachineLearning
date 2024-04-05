import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from warnings import simplefilter


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X.shape, y.shape

#原始影像
# figure = plt.figure(figsize=(10, 10))
# rows, cols = 8, 8
# for i in range(1, rows * cols + 1):
#     # 随机选数据
#     # sample_idx = np.random.randint(0, len(X))
#     # img, label = X.values[sample_idx].reshape(28, 28), y[sample_idx]
#     # 按数据集的顺序选前几个数据
#     img, label = X.values[i - 1].reshape(28, 28), y[i - 1]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis('off')
#     plt.imshow(img, cmap='gray')
# plt.show()

#保留95%的方差，需多少特徵
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)	# 压缩
X_recovered = pca.inverse_transform(X_reduced)
print(pca.n_components_)


#利用主要特徵還原影像
figure = plt.figure(figsize=(10, 10))
rows, cols = 8, 8
for i in range(1, rows * cols + 1):
    # 随机选数据
    # sample_idx = np.random.randint(0, len(X))
    # img, label = X.values[sample_idx].reshape(28, 28), y[sample_idx]
    # 按数据集的顺序选前几个数据
    img, label = X_recovered[i - 1].reshape(28, 28), y[i - 1]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()
