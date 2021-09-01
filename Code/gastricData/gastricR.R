library(samr)
library(reticulate)
use_python("/usr/bin/python3")
source_python("gastricData.py")

# MSI_reshaped_R <- MSI_reshaped
MassSpec_R <- MassSpec
tsne3_R <- tsne3

plot(tsne3_R)

# set.seed(100)
# samfit <- SAM(MassSpec_R, tsne3_R, resp.type = "Two class unpaired")

# # examine significant gene list
# print(samfit)

# # plot results
# plot(samfit)