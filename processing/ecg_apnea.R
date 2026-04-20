library(data.table)
library(plotly)

dt <- fread("ecg_data.csv")

colSums(is.na(dt))
dt <- dt[!is.na(dt$label), ]
#remove signal that isnt labelled
dt[rowSums(is.na(dt)) > 0, patient]
#10 patients with missing data
na_patients <- dt[!complete.cases(dt), patient]
dt <- dt[!patient %in% na_patients]
#drop all patients with na values (we have a lot of data)


Q1 <- quantile(dt$rr_mean, .25)
Q3 <- quantile(dt$rr_mean, .75)
IQR <- IQR(dt$rr_mean)
outliers <- subset(dt, dt$rr_mean<(Q1 - 1.5*IQR) | dt$rr_mean>(Q3 + 1.5*IQR))

dt[, .(.N), patient]


a <- plot_ly(data = dt[patient == "a04"], x = 1:nrow(dt[patient == "a04"]), y = ~rr_mean, color = ~label,
             type = "scatter", mode = "markers", showlegend = T)
a

