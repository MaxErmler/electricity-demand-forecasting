# 562 project
#maximillian ermler, ermler@umich.edu

#libraries
library(readxl)
library(tidyverse)
library(lubridate)
library(forecast)
library(MASS)
library(faraway)
library(lars)
library(quantreg)
library(robustbase)
library(pls)
library(mgcv)
##########################################################################
#DATA SECTION

###################
# #DETROIT + MISO
# #DET weather data
# DET.weather2019 <- read.csv("72537094847 (4).csv")
# DET.weather2020 <- read.csv("72537094847 (1).csv")
# DET.weather2021 <- read.csv("72537094847 (2).csv")
# DET.weather2022 <- read.csv("72537094847 (3).csv")
# DET.weather2023 <- read.csv("72537094847.csv")
# DET.weather2024 <- read.csv("72537094847 (5).csv")
# DET.weather2025 <- read.csv("72537094847 (6).csv")
# library(readxl)
# #MISO power demand
# dat.miso <- read_excel("MISO.xlsx")

###############

###############
##TOL weather data
#TOL.weather2022_2026 <- read.csv("open-meteo-Toledo-1.csv", skip =3)
#TOL.weather2019_2022 <- read.csv("open-meteo-Toledo-2.csv", skip =3)
#TOL power demand
library(readxl)
#TOL.demand <- read_excel("OH_Hourly_Load_by_Class.xlsx")
#TOL merged data
TOL.data <- read.csv("toledo_clean.csv")
###################################################################
#holidays for 2019-2025
holidays <- as.Date(c(
  # New Year's Day
  "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",
  # Memorial Day
  "2019-05-27","2020-05-25", "2021-05-31", "2022-05-30","2023-05-29", "2024-05-27", "2025-05-26",
  # Independence Day
  "2019-07-04", "2020-07-04", "2021-07-04","2022-07-04", "2023-07-04", "2024-07-04", "2025-07-04",
  # Labor Day
  "2019-09-02","2020-09-07", "2021-09-06", "2022-09-05", "2023-09-04","2024-09-02", "2025-09-01",
  # Thanksgiving
  "2019-11-28", "2020-11-26", "2021-11-25", "2022-11-24","2023-11-23", "2024-11-28", "2025-11-27",
  # Christmas
  "2019-12-25","2020-12-25", "2021-12-25", "2022-12-25","2023-12-25", "2024-12-25","2025-12-25"
))
###################################################################
#FEATURE TRANSFORMATON

#convert to datetime to time function
TOL.data$datetime <- as.POSIXct(TOL.data$datetime)

#transform date feature into sine cosine pair
library(lubridate)
TOL.data <- TOL.data %>% 
    mutate(
      'hour sine' = sin(2*pi * hour(datetime)/24),
      'hour cosine' = cos(2*pi * hour(datetime)/24),
      'week sine' = sin(2*pi * wday(datetime)/7),
      'week cosine' = cos(2*pi * wday(datetime)/7),
      'month sine' = sin(2*pi * month(datetime)/12),
      'month cosine' = cos(2*pi* month(datetime)/12),
      'is it a weekend?' = ifelse(wday(datetime) %in% c(1,7), 1,0),
      'year' = year(datetime),
      'is it COVID' = ifelse(datetime >= "2020-03-01" & datetime<= "2021-06-01",1,0),
      'is it a heatwave?' = ifelse(temperature > quantile(temperature,.9) & lag(temperature,24) > quantile(temperature,.9),1,0),
      'is it a holiday?' = ifelse(as.Date(datetime) %in% holidays,1,0),
      'temp_sq' = temperature^2,
      'temp_cb' = temperature^3,
      'lag_day_demand' = lag(demand,24),
      'lag_week_demand' = lag(demand,168)
    )


#####################################################################





#####################################################################
#PLOTS that investigate and justify sine/cosine transformation
# also justifies that we can drop YEAR as a predictor because of training sets.

##############
#investigating Demand VS TIME
#after investigation, median is stable, and thus year is noise at annual level. Predictor can be dropped if chosen.
par(mfrow = c(3, 3))
plot(TOL.data$year,TOL.data$demand, main = "Year vs demand")
boxplot(demand ~ year, data = TOL.data, main = "Year vs demand" )
hist(TOL.data$demand[TOL.data$year == c(2019)], main = "demand 2019")
hist(TOL.data$demand[TOL.data$year == c(2020)], main = "demand 2020")
hist(TOL.data$demand[TOL.data$year == c(2021)], main = "demand 2021")
hist(TOL.data$demand[TOL.data$year == c(2022)],main = "demand 2022")
hist(TOL.data$demand[TOL.data$year == c(2023)],main = "demand 2023")
hist(TOL.data$demand[TOL.data$year == c(2024)],main = "demand 2024")
hist(TOL.data$demand[TOL.data$year == c(2025)], main = "demand 2025")
par(mfrow = c(1, 1))

#pattern consistant across years, 2021 shows more volatility
par(mfrow = c(3, 3))
plot(TOL.data$datetime , TOL.data$demand)
plot(TOL.data$datetime[TOL.data$year == '2019'], TOL.data$demand[TOL.data$year == '2019'])
plot(TOL.data$datetime[TOL.data$year == '2020'], TOL.data$demand[TOL.data$year == '2020'])
plot(TOL.data$datetime[TOL.data$year == '2021'], TOL.data$demand[TOL.data$year == '2021'])
plot(TOL.data$datetime[TOL.data$year == '2022'], TOL.data$demand[TOL.data$year == '2022'])
plot(TOL.data$datetime[TOL.data$year == '2023'], TOL.data$demand[TOL.data$year == '2023'])
plot(TOL.data$datetime[TOL.data$year == '2024'], TOL.data$demand[TOL.data$year == '2024'])
plot(TOL.data$datetime[TOL.data$year == '2025'], TOL.data$demand[TOL.data$year == '2025'])
par(mfrow = c(1, 1))

#seasonal differencing shows correlation with time, cyclical patterns
par(mfrow = c(2, 2))
acf(TOL.data$demand, lag.max =168)
Acf(diff(TOL.data$demand, lag = 24), lag.max = 100)
Acf(diff(TOL.data$demand, lag = 168), lag.max = 400)
#############


##############################################################################
#TRAIN/VALIDATION/TEST Split

train <- TOL.data[TOL.data$year < 2023,]
val <- TOL.data[TOL.data$datetime >= '2023-01-01' & TOL.data$datetime < '2023-11-01',]
test <- TOL.data[TOL.data$datetime >= '2023-11-01',]

# nrow(train) / nrow(TOL.data)
# nrow(val) / nrow(TOL.data)
# nrow(test) / nrow(TOL.data)

#transformation ln(y)
train.logy <- train %>% mutate(log.demand = log(demand))
val.logy <- val %>% mutate(log.demand = log(demand))
test.logy <- test %>% mutate(log.demand = log(demand))


train <- train[!is.na(train$lag_week_demand), ] #clear out 168 NA's
train.logy <- train.logy[!is.na(train.logy$lag_week_demand), ]

#create validation matrix on val dataset
x.val.full <- model.matrix(log.demand ~ . - demand - year - datetime, 
                           data = val.logy)[, -1]
x.val.drop <- model.matrix(log.demand ~ . - demand - year - datetime - precipitation 
                           -apparent_temp - direct_rad_inst - shortwave_rad 
                           - soil_temp - et0 - wind_gusts, 
                           data = val.logy)[, -1]

#create test matrix on the test dataset
x.test.full <- model.matrix(log.demand ~ . - demand - year - datetime, 
                           data = test.logy)[, -1]
x.test.drop <- model.matrix(log.demand ~ . - demand - year - datetime - precipitation 
                           -apparent_temp - direct_rad_inst - shortwave_rad 
                           - soil_temp - et0 - wind_gusts, 
                           data = test.logy)[, -1]

#############################################################################
###########################################################################
#########
# third order temp is best

plot(train.logy$temperature,train.logy$log.demand)
tempcurve <- lm(log.demand ~ poly(temperature,3, raw=TRUE), data = train.logy)
summary(tempcurve)
temp.seq <- seq(min(train.logy$temperature), max(train.logy$temperature), length = 2000)
pred.curve <- predict(tempcurve, newdata = data.frame(temperature = temp.seq))
lines(temp.seq, pred.curve, col = "red", lwd = 2)



##########
##############################
#############OLS on full training data before y transform
model.ols <- lm(demand ~. - datetime - year , data = train)
# summary(model.ols)
# par(mfrow = c(2, 2))
# plot(model.ols)

#box-cox, shows lambda=0, so ln(y) transformation
library(MASS)
# boxcox(model.ols, lambda= seq(-2,2,by=.1))





################ OLS on log(y) , before removing collinear variables, full
model.ols.full <- lm(log.demand ~ . - datetime - year - demand , data = train.logy)
summary(model.ols.full)

vif(model.ols.full)
# par(mfrow = c(2, 2))
# plot(model.ols.full)

plot(train.logy$datetime,residuals(model.ols.full), type ='l')

#predict on validation set
pred.ols.full <- exp(predict(model.ols.full, newdata = val.logy))
RMSE.ols.full.val <- sqrt(mean((val.logy$demand - pred.ols.full)^2))
MAE.ols.full.val <- mean(abs(val.logy$demand - pred.ols.full))

#test on test set
pred.ols.test.full <- exp(predict(model.ols.full, newdata = test.logy))
RMSE.ols.test.full <- sqrt(mean((test.logy$demand - pred.ols.test.full)^2))
MAE.ols.test.full <- mean(abs(test.logy$demand - pred.ols.test.full))





#################### outlier investigation
library(MASS)
stud.resid <- studres(model.ols.full)

n <- nrow(train.logy)
p <- length(coef(model.ols.full))
bonf.threshold <- qt(1 - 0.05/(2*n), n - p - 1)
#bonf.threshold
outlier.idx <- which(abs(stud.resid) > bonf.threshold)
# length(outlier.idx)
# outlier.idx
# train.logy[outlier.idx, c("datetime", "demand", "temperature", "snow_depth", "wind_speed", "precipitation")]

#two outliers out of 35,000+ , on memorial day. leave them in, negligible




##################### WLS full
#inverse variance
weights.full <- 1 / fitted(model.ols.full)^2
model.wls.full <- lm(log.demand ~ . - datetime - year - demand, data = train.logy, weights = weights.full)

# val
pred.wls.val.full <- exp(predict(model.wls.full, newdata = val.logy))
RMSE.wls.val.full <- sqrt(mean((val.logy$demand - pred.wls.val.full)^2))
MAE.wls.val.full <- mean(abs(val.logy$demand - pred.wls.val.full))

# test
pred.wls.test.full <- exp(predict(model.wls.full, newdata = test.logy))
RMSE.wls.test.full <- sqrt(mean((test.logy$demand - pred.wls.test.full)^2))
MAE.wls.test.full <- mean(abs(test.logy$demand - pred.wls.test.full))




############## RIDGE regression full
model.ridge.full <- lm.ridge(log.demand ~ . - datetime - year - demand , data = train.logy, lambda = seq(0, 10, 0.01))
summary(model.ridge.full)
#MASS::select(model.ridge.full)
best.lambda.full <- 0.2
coef.ridge.full <- coef(model.ridge.full)[which(model.ridge.full$lambda == best.lambda.full), ]

#predict on validation set
pred.ridge.log.full <- x.val.full %*% coef.ridge.full[-1] + coef.ridge.full[1]
pred.ridge.demand.full <- exp(pred.ridge.log.full)
RMSE.ridge.val.full <- sqrt(mean((val.logy$demand - pred.ridge.demand.full)^2))
MAE.ridge.val.full <- mean(abs(val.logy$demand - pred.ridge.demand.full))

# RMSE.ridge.val.full
# MAE.ridge.val.full

#test on test set
pred.ridge.log.test.full <- x.test.full %*% coef.ridge.full[-1] + coef.ridge.full[1]
pred.ridge.test.full <- exp(pred.ridge.log.test.full)
RMSE.ridge.test.full <- sqrt(mean((test.logy$demand - pred.ridge.test.full)^2))
MAE.ridge.test.full <- mean(abs(test.logy$demand - pred.ridge.test.full))





############## LASSO regression full
x.train.full <- model.matrix(log.demand ~ . - demand - year - datetime, data = train.logy)[, -1]
y.train <- train.logy$log.demand

model.lasso.full <- lars(x.train.full, y.train, type = "lasso")
best.step.cp.full <- which.min(model.lasso.full$Cp)
coef.lasso.full <- coef(model.lasso.full, s = best.step.cp.full, mode = "step")

#predict on validation set
pred.lasso.full <- predict(model.lasso.full, x.val.full, s = best.step.cp.full, mode = "step", type = "fit")$fit
pred.demand.lasso.full <- exp(pred.lasso.full)

RMSE.lasso.full.val <- sqrt(mean((val.logy$demand - pred.demand.lasso.full)^2))
MAE.lasso.full.val <- mean(abs(val.logy$demand - pred.demand.lasso.full))
# RMSE.lasso.full.val
# MAE.lasso.full.val

#predict on the test set
pred.lasso.test.full <- predict(model.lasso.full, x.test.full, s = best.step.cp.full, 
                                mode = "step", type = "fit")$fit
pred.demand.lasso.test.full <- exp(pred.lasso.test.full)
RMSE.lasso.test.full <- sqrt(mean((test.logy$demand - pred.demand.lasso.test.full)^2))
MAE.lasso.test.full <- mean(abs(test.logy$demand - pred.demand.lasso.test.full))
# RMSE.lasso.test.full
# MAE.lasso.test.full





############ Huber method full
x.train.full.int <- cbind(intercept = 1, x.train.full)
model.huber.full <- rlm(x = x.train.full.int, y = y.train, method = "M")
coef.huber.full <- coef(model.huber.full)

# val
pred.huber.log.val.full <- cbind(1, x.val.full) %*% coef.huber.full
pred.huber.demand.val.full <- exp(pred.huber.log.val.full)
RMSE.huber.val.full <- sqrt(mean((val.logy$demand - pred.huber.demand.val.full)^2))
MAE.huber.val.full <- mean(abs(val.logy$demand - pred.huber.demand.val.full))

# test
pred.huber.log.test.full <- cbind(1, x.test.full) %*% coef.huber.full
pred.huber.demand.test.full <- exp(pred.huber.log.test.full)
RMSE.huber.test.full <- sqrt(mean((test.logy$demand - pred.huber.demand.test.full)^2))
MAE.huber.test.full <- mean(abs(test.logy$demand - pred.huber.demand.test.full))





############ LAD full
model.lad.full <- rq(log.demand ~ . - demand - year - datetime, data = train.logy, tau = 0.5)

pred.lad.val.full <- exp(predict(model.lad.full, newdata = val.logy))
RMSE.lad.val.full <- sqrt(mean((val.logy$demand - pred.lad.val.full)^2))
MAE.lad.val.full <- mean(abs(val.logy$demand - pred.lad.val.full))

pred.lad.test.full <- exp(predict(model.lad.full, newdata = test.logy))
RMSE.lad.test.full <- sqrt(mean((test.logy$demand - pred.lad.test.full)^2))
MAE.lad.test.full <- mean(abs(test.logy$demand - pred.lad.test.full))





############ LTS full
model.lts.full <- ltsReg(log.demand ~ . - demand - year - datetime, data = train.logy)
coef.lts.full <- coef(model.lts.full)

# validation
pred.lts.log.val.full <- cbind(1, x.val.full) %*% coef.lts.full
pred.lts.demand.val.full <- exp(pred.lts.log.val.full)
RMSE.lts.val.full <- sqrt(mean((val.logy$demand - pred.lts.demand.val.full)^2))
MAE.lts.val.full <- mean(abs(val.logy$demand - pred.lts.demand.val.full))

# test
pred.lts.log.test.full <- cbind(1, x.test.full) %*% coef.lts.full
pred.lts.demand.test.full <- exp(pred.lts.log.test.full)
RMSE.lts.test.full <- sqrt(mean((test.logy$demand - pred.lts.demand.test.full)^2))
MAE.lts.test.full <- mean(abs(test.logy$demand - pred.lts.demand.test.full))




############# PCR full (only doing full)
library(pls)
model.pcr.full <- pcr(log.demand ~ . - demand - year - datetime, 
                      data = train.logy, 
                      scale = TRUE,     
                      validation = "CV") 
summary(model.pcr.full)
ncomp.pcr <- 25

#pred on val
pred.pcr.val <- exp(predict(model.pcr.full, newdata = val.logy, ncomp = ncomp.pcr))
RMSE.pcr.val <- sqrt(mean((val.logy$demand - pred.pcr.val)^2))
MAE.pcr.val <- mean(abs(val.logy$demand - pred.pcr.val))

#pred on test
pred.pcr.test <- exp(predict(model.pcr.full, newdata = test.logy, ncomp = ncomp.pcr))
RMSE.pcr.test <- sqrt(mean((test.logy$demand - pred.pcr.test)^2))
MAE.pcr.test <- mean(abs(test.logy$demand - pred.pcr.test))





##############################################################
################## drop collinearity
#VIF
# vif(model.ols.full)
#drop variables: precipitation, apparent_temp, direct_rad_inst, shortwave_rad, soil_temp, et0, wind_gusts




################# OLS without the collinear terms , dropped
model.ols.drop <- lm(log.demand ~ . -demand - year - datetime - precipitation 
                  - apparent_temp - direct_rad_inst - shortwave_rad - soil_temp 
                  - et0 - wind_gusts, data = train.logy)
# summary(model.ols.drop)
# #VIF
# vif(model.ols.drop)

#predict on validation set
pred.ols.val.drop <- exp(predict(model.ols.drop, newdata = val.logy))
RMSE.ols.val.drop <- sqrt(mean((val.logy$demand - pred.ols.val.drop)^2))
MAE.ols.val.drop <- mean(abs(val.logy$demand - pred.ols.val.drop))

# RMSE.ols.val.drop
# MAE.ols.val.drop

#test on test set
pred.ols.test.drop <- exp(predict(model.ols.drop, newdata = test.logy))
RMSE.ols.test.drop <- sqrt(mean((test.logy$demand - pred.ols.test.drop)^2))
MAE.ols.test.drop <- mean(abs(test.logy$demand - pred.ols.test.drop))




############ WLS drop
weights.drop <- 1 / fitted(model.ols.drop)^2

model.wls.drop <- lm(log.demand ~ . - demand - year - datetime - precipitation 
                     - apparent_temp - direct_rad_inst - shortwave_rad 
                     - soil_temp - et0 - wind_gusts, 
                     data = train.logy, weights = weights.drop)

# val
pred.wls.val.drop <- exp(predict(model.wls.drop, newdata = val.logy))
RMSE.wls.val.drop <- sqrt(mean((val.logy$demand - pred.wls.val.drop)^2))
MAE.wls.val.drop <- mean(abs(val.logy$demand - pred.wls.val.drop))

# test
pred.wls.test.drop <- exp(predict(model.wls.drop, newdata = test.logy))
RMSE.wls.test.drop <- sqrt(mean((test.logy$demand - pred.wls.test.drop)^2))
MAE.wls.test.drop <- mean(abs(test.logy$demand - pred.wls.test.drop))





################# LASSO regression dropped
library(lars)
x.train.drop <- model.matrix(log.demand ~ . - demand - year - datetime - precipitation 
                        - apparent_temp - direct_rad_inst - shortwave_rad 
                        - soil_temp - et0 - wind_gusts, 
                        data = train.logy)[, -1]
y.train <- train.logy$log.demand

model.lasso.drop <- lars(x.train.drop,y.train, type = "lasso")
best.step.cp.drop <- which.min(model.lasso.drop$Cp)
coef.lasso.drop <- coef(model.lasso.drop, s = best.step.cp.drop, mode = "step")

#predict on validation set
pred.lasso.drop <- predict(model.lasso.drop, x.val.drop, s = best.step.cp.drop, mode='step', type='fit')$fit
pred.demand.lasso.drop <- exp(pred.lasso.drop)

RMSE.lasso.val.drop <- sqrt(mean((val.logy$demand - pred.demand.lasso.drop)^2))
MAE.lasso.val.drop <- mean(abs(val.logy$demand - pred.demand.lasso.drop))

# RMSE.lasso.val.drop
# MAE.lasso.val.drop

#test on test set
test.lasso.drop <- predict(model.lasso.drop, x.test.drop, s = best.step.cp.drop, mode='step', type='fit')$fit
test.demand.lasso.drop <- exp(test.lasso.drop)

RMSE.lasso.test.drop <- sqrt(mean((test.logy$demand - test.demand.lasso.drop)^2))
MAE.lasso.test.drop <- mean(abs(test.logy$demand - test.demand.lasso.drop))

# RMSE.lasso.test.drop
# MAE.lasso.test.drop





############## RIDGE regression dropped
model.ridge.drop <- lm.ridge(log.demand ~ . - demand - year - datetime 
                             - precipitation - apparent_temp - direct_rad_inst 
                             - shortwave_rad - soil_temp - et0 - wind_gusts, 
                             data = train.logy, lambda = seq(0, 10, 0.01))
#MASS::select(model.ridge.drop)
best.lambda.drop <- 2.63
coef.ridge.drop <- coef(model.ridge.drop)[which(model.ridge.drop$lambda == best.lambda.drop), ]

#predict on validation set
pred.ridge.log.drop <- x.val.drop %*% coef.ridge.drop[-1] + coef.ridge.drop[1]
pred.ridge.demand.drop <- exp(pred.ridge.log.drop)
RMSE.ridge.val.drop <- sqrt(mean((val.logy$demand - pred.ridge.demand.drop)^2))
MAE.ridge.val.drop <- mean(abs(val.logy$demand - pred.ridge.demand.drop))

# RMSE.ridge.val.drop
# MAE.ridge.val.drop

#test on test set
pred.ridge.log.test.drop <- x.test.drop %*% coef.ridge.drop[-1] + coef.ridge.drop[1]
pred.ridge.test.drop <- exp(pred.ridge.log.test.drop)
RMSE.ridge.test.drop <- sqrt(mean((test.logy$demand - pred.ridge.test.drop)^2))
MAE.ridge.test.drop <- mean(abs(test.logy$demand - pred.ridge.test.drop))





############ Huber method drop
x.train.drop.int <- cbind(intercept = 1, x.train.drop)
model.huber.drop <- rlm(x = x.train.drop.int, y = y.train, method = "M")
coef.huber.drop <- coef(model.huber.drop)

# val
pred.huber.log.val.drop <- cbind(1, x.val.drop) %*% coef.huber.drop
pred.huber.demand.val.drop <- exp(pred.huber.log.val.drop)
RMSE.huber.val.drop <- sqrt(mean((val.logy$demand - pred.huber.demand.val.drop)^2))
MAE.huber.val.drop <- mean(abs(val.logy$demand - pred.huber.demand.val.drop))

# test
pred.huber.log.test.drop <- cbind(1, x.test.drop) %*% coef.huber.drop
pred.huber.demand.test.drop <- exp(pred.huber.log.test.drop)
RMSE.huber.test.drop <- sqrt(mean((test.logy$demand - pred.huber.demand.test.drop)^2))
MAE.huber.test.drop <- mean(abs(test.logy$demand - pred.huber.demand.test.drop))




############ LAD drop
model.lad.drop <- rq(log.demand ~ . - demand - year - datetime 
                     - precipitation - apparent_temp - direct_rad_inst 
                     - shortwave_rad - soil_temp - et0 - wind_gusts,
                     data = train.logy, tau = 0.5)

pred.lad.val.drop <- exp(predict(model.lad.drop, newdata = val.logy))
RMSE.lad.val.drop <- sqrt(mean((val.logy$demand - pred.lad.val.drop)^2))
MAE.lad.val.drop <- mean(abs(val.logy$demand - pred.lad.val.drop))

pred.lad.test.drop <- exp(predict(model.lad.drop, newdata = test.logy))
RMSE.lad.test.drop <- sqrt(mean((test.logy$demand - pred.lad.test.drop)^2))
MAE.lad.test.drop <- mean(abs(test.logy$demand - pred.lad.test.drop))





############ LTS drop
model.lts.drop <- ltsReg(log.demand ~ . - demand - year - datetime 
                         - precipitation - apparent_temp - direct_rad_inst 
                         - shortwave_rad - soil_temp - et0 - wind_gusts,
                         data = train.logy)

coef.lts.drop <- coef(model.lts.drop)

# validation
pred.lts.log.val.drop <- cbind(1, x.val.drop) %*% coef.lts.drop
pred.lts.demand.val.drop <- exp(pred.lts.log.val.drop)
RMSE.lts.val.drop <- sqrt(mean((val.logy$demand - pred.lts.demand.val.drop)^2))
MAE.lts.val.drop <- mean(abs(val.logy$demand - pred.lts.demand.val.drop))

# test
pred.lts.log.test.drop <- cbind(1, x.test.drop) %*% coef.lts.drop
pred.lts.demand.test.drop <- exp(pred.lts.log.test.drop)
RMSE.lts.test.drop <- sqrt(mean((test.logy$demand - pred.lts.demand.test.drop)^2))
MAE.lts.test.drop <- mean(abs(test.logy$demand - pred.lts.demand.test.drop))




################## ARIMA model , forcing 24 as a pattern from ACF plot

demand.ts <- ts(train.logy$log.demand, frequency = 24) # from acf plot 24 is significant
model.arima <- auto.arima(demand.ts,stepwise = TRUE,approximation = TRUE,trace = TRUE)


pred.arima <- forecast(model.arima, h = nrow(test.logy))
pred.arima.demand <- exp(pred.arima$mean)

RMSE.arima <- sqrt(mean((test.logy$demand - pred.arima.demand)^2))
MAE.arima <- mean(abs(test.logy$demand - pred.arima.demand))

cat("SARIMA(1,0,3)(2,1,0)[24] Test RMSE:", round(RMSE.arima, 1), "\n")
cat("SARIMA(1,0,3)(2,1,0)[24] Test MAE:", round(MAE.arima, 1), "\n")



################### SARIMAX
model.sarimax <- readRDS("model_sarimax.rds")
library(forecast)
#model.sarimax <- Arima(demand.ts,order = c(1, 0, 3), seasonal = list(order = c(2, 1, 0), period = 24),xreg = x.train.full)
pred.sarimax <- forecast(model.sarimax, h = nrow(test.logy), xreg = x.test.full)

pred.sarimax.demand <- exp(pred.sarimax$mean)
RMSE.sarimax <- sqrt(mean((test.logy$demand - pred.sarimax.demand)^2))
MAE.sarimax <- mean(abs(test.logy$demand - pred.sarimax.demand))

cat("SARIMAX Test RMSE:", round(RMSE.sarimax, 1), "\n")
cat("SARIMAX Test MAE:",  round(MAE.sarimax, 1),  "\n")




# ################### #### #############.    RESULTS 


cat("VALIDATION RESULTS:\n")
cat("Full Model:\n")
cat("OLS   RMSE =", round(RMSE.ols.full.val, 1),   "MAE =", round(MAE.ols.full.val, 1), "\n")
cat("Ridge RMSE =", round(RMSE.ridge.val.full, 1),  "MAE =", round(MAE.ridge.val.full, 1), "\n")
cat("LASSO RMSE =", round(RMSE.lasso.full.val, 1),  "MAE =", round(MAE.lasso.full.val, 1), "\n")
cat("PCR   RMSE =", round(RMSE.pcr.val, 1),         "MAE =", round(MAE.pcr.val, 1), "\n")
cat("Huber RMSE =", round(RMSE.huber.val.full, 1),  "MAE =", round(MAE.huber.val.full, 1), "\n")
cat("LAD   RMSE =", round(RMSE.lad.val.full, 1),    "MAE =", round(MAE.lad.val.full, 1), "\n")
cat("LTS   RMSE =", round(RMSE.lts.val.full, 1),    "MAE =", round(MAE.lts.val.full, 1), "\n")
cat("WLS   RMSE =", round(RMSE.wls.val.full, 1),  "MAE =", round(MAE.wls.val.full, 1), "\n")
cat("Dropped Model:\n")
cat("OLS   RMSE =", round(RMSE.ols.val.drop, 1),    "MAE =", round(MAE.ols.val.drop, 1), "\n")
cat("Ridge RMSE =", round(RMSE.ridge.val.drop, 1),  "MAE =", round(MAE.ridge.val.drop, 1), "\n")
cat("LASSO RMSE =", round(RMSE.lasso.val.drop, 1),  "MAE =", round(MAE.lasso.val.drop, 1), "\n")
cat("Huber RMSE =", round(RMSE.huber.val.drop, 1),  "MAE =", round(MAE.huber.val.drop, 1), "\n")
cat("LAD   RMSE =", round(RMSE.lad.val.drop, 1),    "MAE =", round(MAE.lad.val.drop, 1), "\n")
cat("LTS   RMSE =", round(RMSE.lts.val.drop, 1),    "MAE =", round(MAE.lts.val.drop, 1), "\n")
cat("WLS   RMSE =", round(RMSE.wls.val.drop, 1),  "MAE =", round(MAE.wls.val.drop, 1), "\n")

cat("\nTEST RESULTS:\n")
cat("Full Model:\n")
cat("OLS   RMSE =", round(RMSE.ols.test.full, 1),   "MAE =", round(MAE.ols.test.full, 1), "\n")
cat("Ridge RMSE =", round(RMSE.ridge.test.full, 1),  "MAE =", round(MAE.ridge.test.full, 1), "\n")
cat("LASSO RMSE =", round(RMSE.lasso.test.full, 1),  "MAE =", round(MAE.lasso.test.full, 1), "\n")
cat("PCR   RMSE =", round(RMSE.pcr.test, 1),         "MAE =", round(MAE.pcr.test, 1), "\n")
cat("Huber RMSE =", round(RMSE.huber.test.full, 1),  "MAE =", round(MAE.huber.test.full, 1), "\n")
cat("LAD   RMSE =", round(RMSE.lad.test.full, 1),    "MAE =", round(MAE.lad.test.full, 1), "\n")
cat("LTS   RMSE =", round(RMSE.lts.test.full, 1),    "MAE =", round(MAE.lts.test.full, 1), "\n")
cat("WLS   RMSE =", round(RMSE.wls.test.full, 1), "MAE =", round(MAE.wls.test.full, 1), "\n")
cat("Dropped Model:\n")
cat("OLS   RMSE =", round(RMSE.ols.test.drop, 1),    "MAE =", round(MAE.ols.test.drop, 1), "\n")
cat("Ridge RMSE =", round(RMSE.ridge.test.drop, 1),  "MAE =", round(MAE.ridge.test.drop, 1), "\n")
cat("LASSO RMSE =", round(RMSE.lasso.test.drop, 1),  "MAE =", round(MAE.lasso.test.drop, 1), "\n")
cat("Huber RMSE =", round(RMSE.huber.test.drop, 1),  "MAE =", round(MAE.huber.test.drop, 1), "\n")
cat("LAD   RMSE =", round(RMSE.lad.test.drop, 1),    "MAE =", round(MAE.lad.test.drop, 1), "\n")
cat("LTS   RMSE =", round(RMSE.lts.test.drop, 1),    "MAE =", round(MAE.lts.test.drop, 1), "\n")
cat("WLS   RMSE =", round(RMSE.wls.test.drop, 1), "MAE =", round(MAE.wls.test.drop, 1), "\n")


cat("\nTime Series Models:\n")
cat("SARIMA  Test RMSE =", round(RMSE.arima, 1),   "MAE =", round(MAE.arima, 1), "\n")
cat("SARIMAX Test RMSE =", round(RMSE.sarimax, 1), "MAE =", round(MAE.sarimax, 1), "\n")






######################################################################
# PREDICTION INTERVALS AND PICP

# get prediction intervals on log scale then back-transform
pi.log <- predict(model.ols.full, newdata = test.logy, 
                  interval = "prediction", level = 0.95)
pi.orig <- exp(pi.log)

# overall PICP
picp <- mean(test.logy$demand >= pi.orig[,"lwr"] & 
               test.logy$demand <= pi.orig[,"upr"])
pi.width <- mean(pi.orig[,"upr"] - pi.orig[,"lwr"])

cat("Overall PICP:", round(picp, 3), "\n")
cat("Mean PI width:", round(pi.width, 0), "KW\n")

# PICP by season
test.logy$season <- ifelse(month(test.logy$datetime) %in% c(12,1,2), "winter",
                           ifelse(month(test.logy$datetime) %in% 3:5,  "spring",
                                  ifelse(month(test.logy$datetime) %in% 6:8, "summer", "fall")))

cat("\nPICP by season:\n")
for(s in c("winter","spring","summer","fall")){
  ii <- test.logy$season == s
  p <- mean(test.logy$demand[ii] >= pi.orig[ii,"lwr"] & 
              test.logy$demand[ii] <= pi.orig[ii,"upr"])
  w <- mean(pi.orig[ii,"upr"] - pi.orig[ii,"lwr"])
  cat(s, "PICP:", round(p,3), " width:", round(w,0), " n=", sum(ii), "\n")
}

# PICP by weather extremes
hot  <- test.logy$temperature > 90
cold <- test.logy$temperature < 10
norm <- test.logy$temperature >= 30 & test.logy$temperature <= 70

cat("\nPICP by weather condition:\n")
if(sum(hot) > 0) cat("Hot (>90F) PICP:", round(mean(test.logy$demand[hot]  >= pi.orig[hot,"lwr"]  & test.logy$demand[hot]  <= pi.orig[hot,"upr"]),3),  " n=", sum(hot), "\n")
if(sum(cold) > 0) cat("Cold (<10F) PICP:", round(mean(test.logy$demand[cold] >= pi.orig[cold,"lwr"] & test.logy$demand[cold] <= pi.orig[cold,"upr"]),3), " n=", sum(cold), "\n")
cat("Normal PICP:", round(mean(test.logy$demand[norm] >= pi.orig[norm,"lwr"] & test.logy$demand[norm] <= pi.orig[norm,"upr"]),3), " n=", sum(norm), "\n")

# PICP during holidays
cat("\nPICP by event type:\n")
hol <- test.logy$`is it a holiday?` == 1
covid <- test.logy$`is it COVID` == 1
heat <- test.logy$`is it a heatwave?` == 1

if(sum(hol) > 0) cat("Holidays PICP:", round(mean(test.logy$demand[hol] >= pi.orig[hol,"lwr"]   & test.logy$demand[hol]   <= pi.orig[hol,"upr"]),3),   " n=", sum(hol), "\n")
if(sum(covid) > 0) cat("COVID period  PICP:", round(mean(test.logy$demand[covid] >= pi.orig[covid,"lwr"] & test.logy$demand[covid] <= pi.orig[covid,"upr"]),3), " n=", sum(covid), "\n")
if(sum(heat) > 0) cat("Heatwave PICP:", round(mean(test.logy$demand[heat] >= pi.orig[heat,"lwr"]  & test.logy$demand[heat]  <= pi.orig[heat,"upr"]),3),  " n=", sum(heat), "\n")












