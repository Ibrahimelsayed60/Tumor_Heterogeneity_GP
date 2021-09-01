import h5py
import numpy as np
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# gc_data = h5py.File('Data/gastricData/gastricData.mat', 'r')

# pixels = np.array(gc_data['goodlist']).T.astype(int)
# spectra = np.where(pixels == 1)[0]

# MSI = np.array(gc_data['MSI_data_cube']).T
# MSI_reshaped = np.reshape(MSI, (443 * 1653, 82), order = 'F')
# MassSpec = MSI_reshaped[spectra, :]

# tsne = TSNE(n_components = 3, random_state = 123, verbose = 1, init = 'pca').fit_transform(MassSpec)
tsne3 = np.load("DataToBeLoaded/tsne3new.npy",allow_pickle=True)
print(tsne3.shape)
# K_Means = KMeans(n_clusters = 3, random_state = 0).fit(tsne3)
kmeans = KMeans(n_clusters = 3, random_state = 0).fit_predict(tsne3)
# labels = (np.reshape(K_Means.labels_, (54833, 1), order = 'F')) + 1
# print(labels.shape)
#np.save("kmeans_test.npy", K_Means)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection = '3d')
plt.title("t-SNE of All Spectra Data with labels and Number of Components is 3 in Scatter Space", loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
ax.set_xlabel("t-SNE dim1")
ax.set_ylabel("t-SNE dim2")
ax.set_zlabel("t-SNE dim3")
ax.scatter(tsne3[kmeans==0, 0], tsne3[kmeans==0, 1], tsne3[kmeans==0, 2]) # , c='red', label ='Cluster 1')
ax.scatter(tsne3[kmeans==1, 0], tsne3[kmeans==1, 1], tsne3[kmeans==1, 2]) # , c='blue', label ='Cluster 2')
ax.scatter(tsne3[kmeans==2, 0], tsne3[kmeans==2, 1], tsne3[kmeans==2, 2]) # , c='green', label ='Cluster 3')
plt.show()
# K_means = np.load("kmeans_test.npy", allow_pickle = True)
# print(K_means.shape)

# fig = plt.figure(figsize = (20, 20))
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(*zip(*tsne3))
# plt.show()

