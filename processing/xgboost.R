packages <- c(
  "data.table",
  "plotly",
  "caret",
  "ranger",
  "corrplot",
  "pROC",
  "xgboost"
)

install.packages(packages)


library(data.table)
library(plotly)
library(caret)
library(ranger)
library(corrplot)
library(pROC)
library(xgboost)

dt <- fread("ecg_data.csv") 
colSums(is.na(dt))

unique(dt[, patient]) 
dt <- dt[!is.na(label)] 
unique(dt[, patient])

#some lag features of the features with the most contributions
dt[, `:=`(
  lag1_feat1 = shift(vlf, 1),
  lag2_feat1 = shift(vlf, 2),
  
  roll3_mean = frollmean(vlf, 3, align = "right"),
  roll3_sd   = frollapply(vlf, 3, sd, align = "right"),
  
  roll5_mean = frollmean(vlf, 5, align = "right"),
  roll5_sd   = frollapply(vlf, 5, sd, align = "right"),
  
  delta1_feat1 = vlf - shift(vlf, 1),
  delta5_feat1 = vlf - shift(vlf, 5),
  
  lag1_feat2 = shift(rr_sd, 1),
  lag2_feat2 = shift(rr_sd, 2),
  
  roll3_mean_2 = frollmean(rr_sd, 3, align = "right"),
  roll3_sd_2   = frollapply(rr_sd, 3, sd, align = "right"),
  
  roll5_mean_2 = frollmean(rr_sd, 5, align = "right"),
  roll5_sd_2   = frollapply(rr_sd, 5, sd, align = "right"),
  
  delta1_feat2 = rr_sd - shift(rr_sd, 1),
  delta5_feat2 = rr_sd - shift(rr_sd, 5),
  
  lag1_feat3 = shift(bpm, 1),
  lag2_feat3 = shift(bpm, 2),
  
  roll3_mean_3 = frollmean(bpm, 3, align = "right"),
  roll3_sd_3   = frollapply(bpm, 3, sd, align = "right"),
  
  
  roll5_mean_3 = frollmean(bpm, 5, align = "right"),
  roll5_sd_3   = frollapply(bpm, 5, sd, align = "right"),
  
  delta1_feat3 = bpm - shift(bpm, 1),
  delta5_feat3 = bpm - shift(bpm, 5),
  
  lag1_feat4 = shift(hf, 1),
  lag2_feat4 = shift(hf, 2),
  
  roll3_mean_4 = frollmean(hf, 3, align = "right"),
  roll3_sd_4   = frollapply(hf, 3, sd, align = "right"),
  
  roll5_mean_4 = frollmean(hf, 5, align = "right"),
  roll5_sd_4   = frollapply(hf, 5, sd, align = "right"),
  
  delta1_feat4 = hf - shift(hf, 1),
  delta5_feat4 = hf - shift(hf, 5)
  
), by = patient]
dt <- na.omit(dt)

train_patients <- c("a01", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", "a11", 
                    "a12", "a13", "a14", "a15", "a16", "b01", "c03", "c05", "c06")

test_patients <- c( "a17", "a18", "a19", "a20", "b03", "c07", "c08")

training <- dt[patient %in% train_patients]
test <- dt[patient %in% test_patients]

print(training[, .N, by = label])
print(test[, .N, by = label])

feature_cols <- setdiff(
  names(training),
  c("label", "patient", "title")
)


train_x <- data.matrix(training[, ..feature_cols])
train_y <- training$label

test_x <- data.matrix(test[, ..feature_cols])
test_y <- test$label

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#XGBOOST

#parameter tuning
'''
xgb_grid <- expand.grid(
  max_depth = c(2,3,4,5,6), 
  eta = c(0.01,0.03,0.05,0.1),
  min_child_weight = c(1,3,5,10), 
  subsample = c(0.7,0.8,1),
  colsample_bytree = c(0.6,0.8,1),
  gamma = c(0,0.1,0.5), 
  scale_pos_weight = c(1,2,5,10),
  F1 = 0,
  AUC = 0,
  Recall = 0
)

set.seed(42)

xgb_grid <- xgb_grid[
  sample(nrow(xgb_grid), 100),
]

folds <- groupKFold(training$patient, k = 5)

for(i in 1:nrow(xgb_grid)){
  
  fold_recall <- c()
  fold_auc <- c()
  fold_f1 <- c()
  
  for(f in seq_along(folds)){
    
    val_idx <- folds[[f]]
    
    train_idx <- setdiff(
      seq_len(nrow(train_x)),
      val_idx
    )
    
    dtrain <- xgb.DMatrix(
      train_x[train_idx,],
      label = train_y[train_idx]
    )
    
    dval <- xgb.DMatrix(
      train_x[val_idx,],
      label = train_y[val_idx]
    )
    
    model <- xgb.train(
      data = dtrain,
      
      params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        
        max_depth = xgb_grid$max_depth[i],
        eta = xgb_grid$eta[i],
        min_child_weight = xgb_grid$min_child_weight[i],
        subsample = xgb_grid$subsample[i],
        colsample_bytree = xgb_grid$colsample_bytree[i],
        gamma = xgb_grid$gamma[i],
        scale_pos_weight = xgb_grid$scale_pos_weight[i]
      ),
      
      nrounds = 3000,
      evals = list(train = dtrain,val = dval),
      early_stopping_rounds = 50,
      verbose = 0
    )
    
    probs <- predict(model, newdata = dval)

    preds <- factor(ifelse(probs >= 0.4, "1", "0"),levels = c("0", "1"))
    
    roc_obj <- roc(train_y[val_idx],probs,levels = c("0", "1"),  quiet = TRUE)
    
    cf <- confusionMatrix(preds, as.factor(train_y[val_idx]),positive = "1")
    
    fold_f1 <- c(fold_f1,cf$byClass["F1"])
    
    fold_recall <- c(fold_recall,cf$byClass["Sensitivity"])
    
    fold_auc <- c(fold_auc,auc(roc_obj))
  
  }
  xgb_grid$F1[i] <- mean(fold_f1)
  xgb_grid$AUC[i] <- mean(fold_auc)
  xgb_grid$Recall[i] <- mean(fold_recall)
}


best_params_recall <- xgb_grid[which.max(xgb_grid$Recall),]
best_params_f1 <- xgb_grid[which.max(xgb_grid$F1),]


print(best_params_recall)
print(best_params_f1)

'''


#best found model

x_model <- xgb.train(
  data = xgb_train,
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 5,
    eta = 0.03,
    min_child_weight = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma = 0,
    scale_pos_weight = 1
  ),
  
  nrounds = 400)


mypred <- predict(x_model, newdata = xgb_test)

roc_obj <- roc(as.factor(test_y), mypred, levels = c("0", "1"))

auc_val <- auc(roc_obj)

print(auc_val)

plot(roc_obj, print.auc = TRUE, main = "ROC Curve")

pred_labels <- factor(ifelse(mypred >= 0.35, "1", "0"),levels = c("0", "1")) #below 0.5 to priotitize recall

confusionMatrix(as.factor(pred_labels),as.factor(test_y), mode="prec_recall", positive = "1")


#per patient in tests
test_results <- copy(test)
test_results$pred <- pred_labels

patient_perf <- test_results[, .(
  Accuracy = mean(pred == label)
), by = patient]

print(patient_perf)

