# Tumor Heterogeneity
# Layout
* <a href="#A"> To-Do / Tasks </a>
* <a href="#B"> General Notes / Useful Links </a>
* <a href="#C"> Tasks' Status </a>

<h2 id="A"> To-Do / Tasks </h2>

1. For gastricData: please visualize protein images at the following mass-to-charge (m/z) values:  [3374, 3409, 3445, 4967, and 14021] (Done)
2. Make a presentation on PCA, Apply the PCA on the MSI gastricData (please remember to use "goodlist" to guide you accessing only spectral information without including the background) doing the following:
   * Use Apply PCA to perform dimensionality reduction from 82 to 2 dimensions (i.e. n_components = 2):
      * Use scatter plot to show the reduced data
      * Show the amount of explained variance in each of  those 2 principal components
      * Color each point in the PCA scatter plot using the protein intensity value at m/z 3374
   * Apply the PCA to reduce the MSI data into:
      * 3 dimensions and show the explained variance retained in each of those 3 principal components
      * 5 dimensions and and show the explained variance in each component
3. Make a presentation on t-SNE, Apply t-SNE (n_components = 2) for the result of PCA and give a comment for the output 
4. Understand concept of SAM and apply it using R language
5. For breastData:
   * Similarly to what we have discussed today, please access the MSI data and show protein images at m/z = [4965, 4999, 5067]
   * Show the average spectrum of patient #1 and Patient #30
   * Apply the above described PCA analysis but on the breast cancer data
6. Analysis of papers
7. Add colors of clusters in KMF (Survival Analysis)
8. Visualzation of clusters in scatter space # (k = 2 to k = 5) #
9. SAM for all intensities in clusters

<h2 id="B"> General Notes / Useful Links </h2>

***PCA and t-SNE Algorithms*** <br>
PCA                         |                        T-SNE
:--------------------------:|:----------------------------:
[Look at direct PCA part](https://www.math.uwaterloo.ca/~aghodsib/courses/f06stat890/readings/tutorial_stat890.pdf) | [Paper](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)
[Python examples](https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python) | [Lecture](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw&ab_channel=GoogleTechTalks)

***Survival Analysis (KAPLAN-MEIER Curves)*** <br>
[Link1](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/BS704_Survival_print.html) <br>
[Link2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3932959/)

<h2 id="C"> Tasks' Status </h2>

| Task           | Assigned to   | Current Status (done or in progress) |
|----------------|---------------|----------------|
| [1] | Ibrahim & Mustafa | - [x] done |
| [2] | Donia & Renad | - [x] done |
| [3] | Mariem & Ibrahim | - [x] done |
| [4] | Mustafa & Mariem | - [ ] in progress |
| [5] | Ibrahim | - [ ] in progress |
| [6] | Renad & Donia | - [o] almost done |
| [7] |  | - [ ]  |
| [8] |  | - [ ]  |
| [9] |  | - [ ]  |
