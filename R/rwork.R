#从UCI机器学习数据库中调取数据集，该数据集是乳腺阿
loc <- "http://archive.ics.uci.edu/ml/machine-learning-databases/"           
ds  <- "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
url <- paste(loc, ds, sep="")
breast <- read.table(url, sep=",", header=FALSE, na.strings="?")
##数据整理过程
#添加列/变量名
names(breast) <- c("ID", "clumpThickness", "sizeUniformity",
                   "shapeUniformity", "maginalAdhesion", 
                   "singleEpithelialCellSize", "bareNuclei", 
                   "blandChromatin", "normalNucleoli", "mitosis", "class")
#删除第一列Id值
df <- breast[-1]
#将最后一列转化为因子，并贴上标签
df$class <- factor(df$class, levels=c(2,4), 
                   labels=c("benign", "malignant"))
## 基于重抽样中的   进行抽样
train <- sample(nrow(df), 0.7*nrow(df))
#3/7分
df.train <- df[train,]
#抽取70%为训练集（489）
df.validate <- df[-train,]
#剩余的为验证集（210）
table(df.train$class)
table(df.validate$class)

#逻辑回归
fit.logit <- glm(class~., data=df.train, family=binomial())
#  注意表达式里的点 . 表示表格里除了因变量（class）里的其它所有预测变量，方便的一种写法。
summary(fit.logit)

prob <- predict(fit.logit, df.validate, type="response")
#type="response"得到预测肿瘤为恶性的概率
logit.pred <- factor(prob > .5, levels=c(FALSE, TRUE), 
                     labels=c("benign", "malignant"))
#概率大于0.5为TRUE，被贴上恶性的标签

logit.perf <- table(df.validate$class, logit.pred,
                    dnn=c("Actual", "Predicted"))
logit.perf


#决策树
library(rpart)
dtree <- rpart(class ~ ., data=df.train, method="class",      
               parms=list(split="information"))
dtree$cptable

dtree.pruned <- prune(dtree, cp=.0392) 
#绘制决策树
library(rpart.plot)
prp(dtree.pruned, type = 2, extra = 104,  
    fallen.leaves = TRUE, main="Decision Tree")


dtree.pred <- predict(dtree.pruned, df.validate, type="class")
dtree.perf <- table(df.validate$class, dtree.pred, 
                    dnn=c("Actual", "Predicted"))
dtree.perf


# 随机森林
library(randomForest)
fit.forest <- randomForest(class~., data=df.train,        
                           na.action=na.roughfix,
                           importance=TRUE) 
importance(fit.forest, type=2) 
forest.pred <- predict(fit.forest, df.validate)         
forest.perf <- table(df.validate$class, forest.pred, 
                     dnn=c("Actual", "Predicted"))
forest.perf

#计算评价指标

performance <- function(table, n=2){
  if(!all(dim(table) == c(2,2)))
    stop("Must be a 2 x 2 table")   #一定要是2×2形式的二联表
  tn = table[1,1]  
  fp = table[1,2]
  fn = table[2,1]
  tp = table[2,2]
  #分别对应位置取值
  sensitivity = tp/(tp+fn)
  #计算敏感度
  specificity = tn/(tn+fp)
  #计算特异度
  ppp = tp/(tp+fp)
  #计算正例命中率
  npp = tn/(tn+fn)
  #计算负例命中率
  hitrate = (tp+tn)/(tp+tn+fp+fn)
  #计算准确率
  result <- paste("Sensitivity = ", round(sensitivity, n) ,    #原来n是设置了保留的位数
                  "\nSpecificity = ", round(specificity, n),       # /n  为换行符
                  "\nPositive Predictive Value = ", round(ppp, n),
                  "\nNegative Predictive Value = ", round(npp, n),
                  "\nAccuracy = ", round(hitrate, n), "\n", sep="")
  cat(result)
}

performance(logit.perf)
performance(dtree.perf)
performance(forest.perf)

