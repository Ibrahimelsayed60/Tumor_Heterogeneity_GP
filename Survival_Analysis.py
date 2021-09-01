import lifelines
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcol
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import scipy
import numpy as np
import pandas as pd

C_data = pd.read_csv("Data/gastricData/Clinical_data_Clusters_test2.csv")
# print(C_data.head())

C_data.Surv_time = C_data.Surv_time / 30.4167
# print(C_data.Cluster.value_counts())
cluster1 = C_data.query("Cluster == 1")
cluster2 = C_data.query("Cluster == 2")
cluster3 = C_data.query("Cluster == 3")
# print(cluster1.head())

# sns.countplot(x = C_data.Cluster, color = 'darkblue')
# plt.show()

kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
kmf3 = KaplanMeierFitter()
kmf1.fit(durations = cluster1["Surv_time"], event_observed = cluster1["Surv_status"], label = "cluster1", alpha = 1)
kmf2.fit(durations = cluster2["Surv_time"], event_observed = cluster2["Surv_status"], label = "cluster2", alpha = 1)
kmf3.fit(durations = cluster3["Surv_time"], event_observed = cluster3["Surv_status"], label = "cluster3", alpha = 1)

kmf1.plot()
kmf2.plot()
kmf3.plot()
plt.xlabel("Survival time in months")
plt.ylabel("Probability")
plt.title('KMF')
plt.show()

results = logrank_test(cluster1['Surv_time'], cluster2['Surv_time'], cluster1['Surv_status'], cluster2['Surv_status'])
print(results)

"""
results1 = multivariate_logrank_test(C_data['Surv_time'], C_data['Cluster'], C_data['Surv_status'])
print(results1)

kmf1.plot()
kmf2.plot()
plt.xlabel("Survival time in months")
plt.ylabel("Probability")
plt.title('KMF')
plt.show()

cph = CoxPHFitter()
cph.fit(C_data[['Surv_time', 'Surv_status', 'Cluster']],'Surv_time', 'Surv_status')
print(cph.print_summary())
"""