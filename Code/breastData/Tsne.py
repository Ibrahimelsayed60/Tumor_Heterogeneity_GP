import h5py
import numpy as np
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# gc_data = h5py.File('breastData.mat', 'r')

# pixels = np.array(gc_data['goodlist']).T.astype(int)
# spectra = np.where(pixels == 1)[0]

# MSI = np.array(gc_data['MSI_data_cube']).T
# MSI_reshaped = np.reshape(MSI, (443 * 1653, 82), order = 'F')
# MassSpec = MSI_reshaped[spectra, :]

# tsne = TSNE(n_components = 3, random_state = 123, verbose = 1, init = 'pca').fit_transform(MassSpec)
tsne3 = np.load("breastData/DataToBeLoaded/tsne3new.npy",allow_pickle=True)
print(tsne3.shape)

kmeans = KMeans(n_clusters = 8, random_state = 0).fit_predict(tsne3)

# fig = plt.figure(figsize = (10, 10))
# ax = fig.add_subplot(111, projection = '3d')
# plt.title("t-SNE in Scatter Space", loc = 'center', fontsize = 25)
# ax.set_xlabel("t-SNE dim1")
# ax.set_ylabel("t-SNE dim2")
# ax.set_zlabel("t-SNE dim3")
# # ax.scatter(tsne3[:, 0], tsne3[:, 1], tsne3[:, 2], c = MassSpec[:, 5])
# ax.scatter(*zip(*tsne3), c = MassSpec[:, 5])

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection = '3d')
plt.title("t-SNE of All Spectra Data with labels and Number of Components is 3 in Scatter Space", loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
ax.set_xlabel("t-SNE dim1")
ax.set_ylabel("t-SNE dim2")
ax.set_zlabel("t-SNE dim3")
ax.scatter(tsne3[kmeans==0, 0], tsne3[kmeans==0, 1], tsne3[kmeans==0, 2]) # , c='red', label ='Cluster 1')
ax.scatter(tsne3[kmeans==1, 0], tsne3[kmeans==1, 1], tsne3[kmeans==1, 2]) # , c='blue', label ='Cluster 2')
ax.scatter(tsne3[kmeans==2, 0], tsne3[kmeans==2, 1], tsne3[kmeans==2, 2]) # , c='green', label ='Cluster 3')
ax.scatter(tsne3[kmeans==3, 0], tsne3[kmeans==3, 1], tsne3[kmeans==3, 2])
ax.scatter(tsne3[kmeans==4, 0], tsne3[kmeans==4, 1], tsne3[kmeans==4, 2])
ax.scatter(tsne3[kmeans==5, 0], tsne3[kmeans==5, 1], tsne3[kmeans==5, 2])
ax.scatter(tsne3[kmeans==6, 0], tsne3[kmeans==6, 1], tsne3[kmeans==6, 2])
ax.scatter(tsne3[kmeans==7, 0], tsne3[kmeans==7, 1], tsne3[kmeans==7, 2])




plt.show()