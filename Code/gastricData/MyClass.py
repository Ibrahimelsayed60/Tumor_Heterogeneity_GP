################################ Section1 - Importing Libraries ################################
from gastricDataLibrary import *

# creating MyClass class
class MyClass:
    #__init__ constructor
    def __init__(self, path):
        self.MyData = h5py.File(path, 'r')
        print(self.MyData.keys())
    
    # methods
    def set_data(self):
        self.x = np.array(self.MyData['x']).T.astype(int)
        self.y = np.array(self.MyData['y']).T.astype(int)
        self.z = np.array(self.MyData['z']).T.astype(int)
        self.peak_list = np.array(self.MyData['peak_list']).T
        self.mz_val = self.peak_list[:, 0]
        self.avg_int = self.peak_list[:, 1]
        self.low_bod = self.peak_list[:, 2]
        self.upp_bod = self.peak_list[:, 3]
        self.ProteinNumber = self.peak_list.shape[0]
        self.pixels = np.array(self.MyData['goodlist']).T.astype(int)
        self.spectra = np.where(self.pixels == 1)[0]
        self.pixel_to_sample_ID = np.array(self.MyData['pixel_to_sample_ID']).T
        self.HE_image = np.array(self.MyData['HE_image']).T
        self.MSI = np.array(self.MyData['MSI_data_cube']).T
        self.reshape_data(self.y[0][0], self.x[0][0])
        # self.plot_images("goodlist", self.pixels_img)
        # self.plot_images("pixel_sample_id", self.pixel_to_sample_ID)
        # self.plot_images("HE_image", self.HE_image)
        # self.plot_MSI()

    def reshape_data(self, height, width):
        self.pixels_img = np.reshape(self.pixels, (height, width), order = 'F')
        self.pixel_sample = np.reshape(self.pixel_to_sample_ID, (height * width, 1), order = 'F')
        self.HE_image_reshaped = np.reshape(self.HE_image, (height * width, 3), order = 'F')
        self.MSI_reshaped = np.reshape(self.MSI, (height * width, self.ProteinNumber), order = 'F')
        self.MassSpec = self.MSI_reshaped[self.spectra, :]

    def print_info(self, argument):
        if(argument == "pixels_info"):
            print("Width of dataset in pixels is: %d, Height is: %d and Number of peaks is: %d" % (self.x, self.y, self.z))
        if(argument == "peak_list"):
            print("Shape of m/z values, average intensity, lower and upper boundary is: %d" % (self.mz_val.shape)) # or self.avg_int.shape or self.low_bod.shape or self.upp_bod.shape
        if(argument == "spectra"):
            print("Number/size of pixels: ", self.pixels.size)
            print("Shape of pixels: ", self.pixels.shape)
            print("Length of non-background data: ", self.spectra.shape[0])
        if(argument == "MSI"):
            print("Shape of MSI: ", self.MSI.shape)
            print("Shape of reshaped MSI data: ", self.MSI_reshaped.shape)
            print("Shape of MSI of only spectra data: ", self.MassSpec.shape)
        if(argument == "PCA"):
            print("PCA with Number of Components = 2 ...")
            print("Shape of PCA for All Spectra Data and Number of Components = 2: ", self.pca_2.shape)
            print("PCA with Number of Components = 3 ...")
            print("Shape of PCA for All Spectra Data and Number of Components = 3: ", self.pca_3.shape)
            print("PCA with Number of Components = 5 ...")
            print("Shape of PCA for All Spectra Data and Number of Components = 5: ", self.pca_5.shape)
            # print("Variances for All Spectra Data and Number of Components = 2: ", self.pca2.explained_variance_ratio_)
            # print("Variances for All Spectra Data and Number of Components = 3: ", self.pca3.explained_variance_ratio_)
            # print("Variances for All Spectra Data and Number of Components = 5: ", self.pca5.explained_variance_ratio_)
        if(argument == "t-SNE"):
            print("t-SNE with Number of Components = 2 ...")
            print("Shape of t-SNE for All Spectra Data and Number of Components = 2: ", self.tsne2.shape)
            print("t-SNE with Number of Components = 3 ...")
            print("Shape of t-SNE for All Spectra Data and Number of Components = 3: ", self.tsne3.shape)

    def plot_images(self, string, variable):
        plt.figure(figsize = (25, 25))
        plt.title(string, loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
        plt.imshow(variable)
        plt.show()

    def plot_MSI(self):
        plt.figure(figsize = (10, 10))
        plt.xlabel("Mass-to-Charge (m/z) Values")
        plt.ylabel("Total Ion Count")
        for i in range(0, self.MassSpec.shape[0]):
            plt.plot(self.mz_val, self.MassSpec[i, :])
        plt.show()

    def dimensionality_reduction(self, technique, n_components):
        if(technique == "PCA"):
            if(n_components == 2):
                self.pca_2 = np.load("Code/gastricData/DataToBeLoaded/pca_2.npy")
                return self.pca_2
            elif(n_components == 3):
                self.pca_3 = np.load("Code/gastricData/DataToBeLoaded/pca_3.npy")
                return self.pca_3
            elif(n_components == 5):
                self.pca_5 = np.load("Code/gastricData/DataToBeLoaded/pca_5.npy")
                return self.pca_5
        if(technique == "t-SNE"):
            if(n_components == 2):
                self.tsne2 = np.load("Code/gastricData/DataToBeLoaded/tsne2.npy")
                return self.tsne2
            elif(n_components == 3):
                self.tsne3 = TSNE(n_components = n_components, random_state = 0, init = "pca").fit_transform(self.MassSpec)
                self.tsne3new = TSNE(n_components = n_components, random_state = 0).fit_transform(self.MassSpec, y = self.tsne3)
                # self.tsne3 = np.load("Code/gastricData/DataToBeLoaded/tsne3.npy")
                # self.tsne3new = np.load("Code/gastricData/DataToBeLoaded/tsne3new.npy")
                return self.tsne3, self.tsne3new

    def plot_dimensionality_reduction(self, argument, string, variable1, variable2):
        if(argument == "PCA2"):
            plt.figure(figsize = (10, 10))
            plt.title(string, loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.scatter(*zip(*variable1), c = variable2) # self.pca_2[:, 0], self.pca_2[:, 1]
            plt.show()
        if(argument == "PCA3"):
            fig = plt.figure(figsize = (10, 10))
            ax = fig.add_subplot(projection = '3d')
            plt.title(string, loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            ax.set_zlabel("PC3")
            plt.scatter(*zip(*variable1), c = variable2) # self.pca_3[:, 0], self.pca_3[:, 1], self.pca_3[:, 2]
            plt.show()
        if(argument == "t-SNE2"):
            plt.figure(figsize = (10, 10))
            plt.title(string, loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
            plt.xlabel('t-SNE dim1')
            plt.ylabel('t-SNE dim2')
            plt.scatter(*zip(*variable1), c = variable2) # self.tsne2[:, 0], self.tsne2[:, 1]
            plt.show()
        if(argument == "t-SNE3"):
            fig = plt.figure(figsize = (10, 10))
            ax = fig.add_subplot(111, projection = '3d')
            plt.title(string, loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
            ax.set_xlabel("t-SNE dim1")
            ax.set_ylabel("t-SNE dim2")
            ax.set_zlabel("t-SNE dim3")
            ax.scatter(*zip(*variable1), c = variable2) # self.tsne3[:, 0], self.tsne3[:, 1], self.tsne3[:, 2]
            plt.show()

    def KMeans_clustering(self, n_clusters):
        self.KMeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(self.tsne3new)
        self.KMeans_labels = np.reshape(self.KMeans.labels_, (54833, 1), order = 'F')
        # self.KMeans_labels = np.load("Code/gastricData/DataToBeLoaded/KMeans_labels.npy")
        self.KMeans_image = np.zeros_like(self.pixels)
        self.KMeans_image[self.spectra, :] = self.KMeans_labels
        self.KMeans_RGB = np.reshape(self.KMeans_image, (self.y[0][0], self.x[0][0]), order = 'F')
        return self.KMeans_labels

    def plot_KMeans_clustering(self): # , string, variable1, variable2):
        default_cmap = cm.get_cmap(lut = 4)
        gnuplot_cmap = cm.get_cmap("gnuplot", lut = 4)
        jet_cmap = cm.get_cmap("jet", lut = 4)

        plt.figure(figsize = (25, 25))
        plt.title('Default K-means Image', loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
        default_cb = plt.imshow(self.KMeans_RGB, cmap = default_cmap)
        cbar = plt.colorbar(default_cb, ticks = [0, 1, 2, 3], shrink = 0.2)
        plt.figure(figsize = (25, 25))
        plt.title('Gnuplot K-means Image', loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
        gnuplot_cb = plt.imshow(self.KMeans_RGB, cmap = gnuplot_cmap)
        cbar = plt.colorbar(gnuplot_cb, ticks = [0, 1, 2, 3], shrink = 0.2)
        plt.figure(figsize = (25, 25))
        plt.title('Jet K-means Image', loc = 'center', fontsize = 25, color = 'red', fontweight = 'bold')
        jet_cb = plt.imshow(self.KMeans_RGB, cmap = jet_cmap)
        cbar = plt.colorbar(jet_cb, ticks = [0, 1, 2, 3], shrink = 0.2)
        plt.show()

    def Clinical_data(self, path):
        self.ClinicalData = pd.read_csv(path)
        print(self.ClinicalData)

    def getPeaks(self, Patient_ID, counter = 0):
        self.indexHeight = np.where(self.pixel_to_sample_ID == Patient_ID)[0]
        self.indexWidth = np.where(self.pixel_to_sample_ID == Patient_ID)[1]
        self.P1_MaxPeak = []
        while(1):
            self.values_P1 = []
            for i, j in zip(self.indexHeight, self.indexWidth):
                self.values_P1.append(self.MSI[i, j, counter])
            self.P1_MaxPeak.append(np.mean(self.values_P1))
            counter += 1
            if counter == self.ProteinNumber:
                break
        self.P1_MaxPeak = np.array(self.P1_MaxPeak)
        np.savetxt("PPP" + str(Patient_ID) + ".csv", self.P1_MaxPeak, delimiter = ",")

    def getClusters(self, Patient_ID, labels):
        # self.kmeans = np.load("kmeans_test.npy")
        # self.labels = (np.reshape(self.kmeans, (54833, 1), order='F')) + 1
        self.clusters = np.zeros_like(self.pixels, order="F")
        self.clusters[self.spectra, :] = labels
        self.clusters_pateints = np.reshape(self.clusters, (443, 1653), order='F')
        self.indexHeight = np.where(self.pixel_to_sample_ID == Patient_ID)[0]
        self.indexWidth = np.where(self.pixel_to_sample_ID == Patient_ID)[1]
        self.cluster1, self.cluster2, self.cluster3, self.total_clusters, self.dominant_cluster1, self.dominant_cluster2, self.dominant_cluster3 = 0, 0, 0, 0, False, False, False

        for i, j in zip(self.indexHeight, self.indexWidth):
            if self.clusters_pateints[i, j] == 1:
                self.cluster1 += 1
            if self.clusters_pateints[i, j] == 2:
                self.cluster2 += 1
            if self.clusters_pateints[i, j] == 3:
                self.cluster3 += 1

        print("cluster1 total number:", self.cluster1)
        print("cluster2 total number:", self.cluster2)
        print("cluster3 total number:", self.cluster3)

        self.total_clusters = (self.cluster1 + self.cluster2 + self.cluster3) / 3
        print("total number of clusters is:", self.total_clusters)
        if self.cluster1 > self.total_clusters:
            self.dominant_cluster1 = True
        if self.cluster2 > self.total_clusters:
            self.dominant_cluster2 = True
        if self.cluster3 > self.total_clusters:
            self.dominant_cluster3 = True

        if self.dominant_cluster1:
            print("Patient" + str(Patient_ID) + " is in cluster1")
        if self.dominant_cluster2:
            print("Patient" + str(Patient_ID) + " is in cluster2")
        if self.dominant_cluster3:
            print("Patient" + str(Patient_ID) + " is in cluster3")

        print("-------------------------------------------------------------")


"""
    PCA function implementation
self.pca2 = PCA(n_components = 2, random_state = 0)
self.pca_2 = self.pca2.fit_transform(self.MassSpec)ww
"""

"""
    t-SNE function implementation
self.tsne3 = TSNE(n_components = 3, random_state = 0).fit_transform(self.MassSpec)
"""

"""
    clustering function implementation
self.KMeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(self.tsne3) # self.tsne3new
self.KMeans_labels = np.reshape(self.KMeans.labels_, (54833, 1), order = 'F')
self.MeanShift = MeanShift(bandwidth = n_clusters).fit(self.tsne3)
"""

"""
kmeans_image = np.zeros_like(pixels, order = 'F')
# print(kmeans_image.shape)
"""

"""
    Saving Data
np.savetxt("pixel_to_sample_ID.csv", self.pixel_to_sample_ID, delimeter = ",")
for i in range(0, 62):
    np.savetxt("proteins_pateints (mz = " + str(i + 1) + ").csv", breastDataVar.MSI[:, :, i], delimiter=",")
"""
