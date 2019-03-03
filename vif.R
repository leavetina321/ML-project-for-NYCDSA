library(car)
setwd('/Users/michaelsankari/Documents/NYC Data Science/Machine Learning Project/github/ML-project-for-NYCDSA')
df <- read.csv('./house-prices-advanced-regression-techniques/cleandata/s2_clean_dummified.csv')

#Remove these columns, ID is not releveant, the other 3 are not in test data
drops <- c('Exterior1st_ImStucc', 'Exterior1st_Stone', 'HouseStyle_2.5Fin', 'Id')
df <- df[, !(names(df) %in% drops)]

fit <- lm(LogSalePrice~., data = df)
summary(fit)
vif(fit) #creates area because one or more variable is linearly dependent on the others

alias(lm(LogSalePrice~., data = df)) #Problem is BsmtCond_None
df$BsmtCond_None <- NULL

fit <- lm(LogSalePrice~., data = df)
summary(fit)

#Do vif and put into data frame that is easier to read
vif_fit <- data.frame(vif(fit))
vif_fit <- data.frame(variable = row.names(vif_fit), vif_score=vif_fit$vif.fit.)
vif_fit$variable <- as.character(vif_fit$variable)
vif_fit <- vif_fit[order(vif_fit$vif_score, decreasing = TRUE),]
row.names(vif_fit) <- 1:nrow(vif_fit)

#Which ones are less than ...
keep <- vif_fit$variable[vif_fit$vif_score<3]
keep <- c(keep, 'LogSalePrice')

df <- df[, (names(df) %in% keep)]

fit2 <- lm(LogSalePrice~., data = df)
summary(fit2)
vif(fit2)

AIC(fit, fit2)
anova(fit2, fit)

write.csv(vif_fit, file = 'vif_scores.csv', row.names = FALSE)
