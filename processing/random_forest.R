packages <- c(
  "data.table",
  "plotly",
  "caret",
  "ranger",
  "corrplot",
  "pROC"
)

install.packages(packages)


library(data.table)
library(plotly)
library(caret)
library(ranger)
library(corrplot)
library(pROC)

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


train_x <- as.data.frame(training[, ..feature_cols])
train_y <- factor(training$label)

test_x <- as.data.frame(test[, ..feature_cols])
test_y <- factor(test$label)

'''
## Parameter testing ----

hyper_grid <- expand.grid(
  ntrees = c(100, 200, 300, 400, 500),
  mtry = c(2,4,6,8,10,12),
  node_size = c(1,3,5,10),
  sample_frac = c(0.6,0.8, 1.0),
  F1 = 0,
  AUC = 0,
  Recall = 0
)

hyper_grid <- hyper_grid[
  hyper_grid$mtry <= (dim(train_x))[2],
] #like 27 post lag features (i kept changing and adding some)

folds <- groupKFold(training$patient, k = 5) #folding by patient so theres separation

for(i in 1:nrow(hyper_grid)){
  
  fold_f1 <- c()
  fold_auc <- c()
  fold_recall <- c()
  
  
  for(f in seq_along(folds)){
    val_idx <- folds[[f]]
    train_idx <- setdiff(
      seq_len(nrow(train_x)),
      val_idx
    )
    
    fold_train_x <- train_x[train_idx, ]
    fold_train_y <- train_y[train_idx]
    fold_val_x<- train_x[val_idx, ]
    fold_val_y <- train_y[val_idx]
    
    model <- ranger(
      x = fold_train_x,
      y = fold_train_y,
      num.trees = hyper_grid$ntrees[i],,
      mtry = hyper_grid$mtry[i],
      min.node.size = hyper_grid$node_size[i],
      probability = TRUE, #yielded better results by getting a probabilty for each class
      importance = "impurity",
      seed = 42,
      sample.fraction = hyper_grid$sample_frac[i]
    )
    
    probs <- predict(model, data= fold_val_x)$predictions[, "1"]
    
    preds <- factor(ifelse(probs >= 0.4, "1", "0"),levels = c("0", "1"))
    
    roc_obj <- roc(fold_val_y,probs,levels = c("0", "1"),  quiet = TRUE)
    
    cf <- confusionMatrix(preds,fold_val_y,positive = "1")
    
    fold_f1 <- c(fold_f1,cf$byClass["F1"])
    
    fold_recall <- c(fold_recall,cf$byClass["Sensitivity"])
    
    fold_auc <- c(fold_auc,auc(roc_obj))
  }
  
  hyper_grid$F1[i] <- mean(fold_f1)
  hyper_grid$AUC[i] <- mean(fold_auc)
  hyper_grid$Recall[i] <- mean(fold_recall)
}


best_params_recall <- hyper_grid[which.max(hyper_grid$Recall),]
best_params_f1 <- hyper_grid[which.max(hyper_grid$F1),]


print(best_params_recall)
print(best_params_f1)

rf_model <- ranger(
  x= train_x,
  y = train_y,
  num.trees = best_params_recall$ntrees,
  mtry = best_params_recall$mtry,
  min.node.size = best_params_recall$node_size,
  probability = TRUE,
  importance = "impurity",
  seed = 42,
  sample.fraction = best_params_recall$sample_frac
)


test_probs <- predict(rf_model,data = test_x)$predictions[, "1"]


roc_obj <- roc(test_y,test_probs,levels = c("0", "1"))

auc_val <- auc(roc_obj)

print(auc_val)

plot(roc_obj,print.auc = TRUE,main = "ROC Curve")


pred_labels <- factor(ifelse(test_probs >= 0.4, "1", "0"),levels = c("0", "1"))

cm <- confusionMatrix(pred_labels,test_y,mode = "prec_recall",positive = "1")

print(cm)

#f1

rf_model <- ranger(
  x= train_x,
  y = train_y,
  num.trees = best_params_f1$ntrees,
  mtry = best_params_f1$mtry,
  min.node.size = best_params_f1$node_size,
  probability = TRUE,
  importance = "impurity",
  seed = 42,
  sample.fraction = best_params_f1$sample_frac
)


test_probs <- predict(rf_model,data = test_x)$predictions[, "1"]


roc_obj <- roc(test_y,test_probs,levels = c("0", "1"))

auc_val <- auc(roc_obj)

print(auc_val)

plot(roc_obj,print.auc = TRUE,main = "ROC Curve")


pred_labels <- factor(ifelse(test_probs >= 0.4, "1", "0"),levels = c("0", "1"))

cm <- confusionMatrix(pred_labels,test_y,mode = "prec_recall",positive = "1")

print(cm)

'''
# actual model

rf_model <- ranger(
  x= train_x,
  y = train_y,
  num.trees = 200,
  mtry = 8,
  min.node.size = 1,
  probability = TRUE,
  importance = "impurity",
  seed = 42,
  sample.fraction = 1
)


test_probs <- predict(rf_model,data = test_x)$predictions[, "1"]


roc_obj <- roc(test_y,test_probs,levels = c("0", "1"))

auc_val <- auc(roc_obj)

print(auc_val)

plot(roc_obj,print.auc = TRUE,main = "ROC Curve")


pred_labels <- factor(ifelse(test_probs >= 0.4, "1", "0"),levels = c("0", "1")) #below 0.5 to priotitize recall

cm <- confusionMatrix(pred_labels,test_y,mode = "prec_recall",positive = "1")

print(cm)



importance_vals <- importance(rf_model)

importance_df <- data.frame(Feature = names(importance_vals),Importance = importance_vals)

importance_df <- importance_df[order(-importance_df$Importance),]

print(head(importance_df, 15))


#per patient in tests
test_results <- copy(test)
test_results$pred <- pred_labels

patient_perf <- test_results[, .(
  Accuracy = mean(pred == label)
), by = patient]

print(patient_perf)

