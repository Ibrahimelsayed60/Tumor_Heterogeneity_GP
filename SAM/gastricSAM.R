library(survival)
library(samr)

set.seed(100)
setwd("C:\\Users\\Mustafa Yehia\\Downloads\\GP\\SAM")

joindata <- read.table("AveragePeaks.txt", sep = "\t", header = TRUE)
msdata <- joindata[,2:83]
# print(msdata)

d = list(x = t(msdata), y = joindata$Surv_status, geneid = as.character(1:82), genenames = names(msdata), logged2 = TRUE) 
# print(d)

samr.obj = samr(d, resp.type = "Two class unpaired", nperms = 200)
# # samr.obj = samr(d, resp.type = "Multiclass", nperms = 200)
delta.table <- samr.compute.delta.table(samr.obj)

delta = 0.8785628882     #Note: Recheck this value by looking at the values of delta.table and choose the value correspondes to (FDR <0.001)

samr.plot(samr.obj,delta)
siggenes.table <- samr.compute.siggenes.table(samr.obj, delta, d, delta.table)
siggenes.table
