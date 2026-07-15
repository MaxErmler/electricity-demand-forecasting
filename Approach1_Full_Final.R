# IOE 562 - Toledo electricity demand
# Approach 1, COVID kept with flag

library(MASS)
library(faraway)
library(lubridate)
library(splines)
library(car)
library(boot)


setwd("C:/Users/hamza/OneDrive/Documents/IOE Mse/Win26/IOE562 Data Science/Project/Toledo data")

dat <- read.csv("toledo_clean.csv")
dat$datetime <- as.POSIXct(dat$datetime, format = "%Y-%m-%d %H:%M:%S")

# COVID period as a flag instead of dropping it
dat$covid_flag <- ifelse(dat$datetime >= as.POSIXct("2020-03-15") &
                         dat$datetime <= as.POSIXct("2021-06-30"), 1, 0)
cat("Rows:", nrow(dat), " COVID rows:", sum(dat$covid_flag), "\n")


# ---------- time features ----------
dat$hour <- factor(hour(dat$datetime))
dat$dow  <- factor(wday(dat$datetime, label = FALSE))
dat$year <- year(dat$datetime)

mnth <- month(dat$datetime)
dat$season <- factor(ifelse(mnth %in% c(12,1,2), "winter",
                     ifelse(mnth %in% 3:5, "spring",
                     ifelse(mnth %in% 6:8, "summer", "fall"))))

holidays <- as.Date(c(
  "2019-01-01","2022-01-01","2023-01-01","2024-01-01","2025-01-01",
  "2019-05-27","2022-05-30","2023-05-29","2024-05-27","2025-05-26",
  "2019-07-04","2022-07-04","2023-07-04","2024-07-04","2025-07-04",
  "2019-09-02","2022-09-05","2023-09-04","2024-09-02","2025-09-01",
  "2019-11-28","2022-11-24","2023-11-23","2024-11-28","2025-11-27",
  "2019-12-25","2022-12-25","2023-12-25","2024-12-25","2025-12-25"))

dat$holiday <- ifelse(as.Date(dat$datetime) %in% holidays, 1, 0)



# chronological split
train <- dat[dat$year < 2023, ]
val   <- dat[dat$datetime >= "2023-01-01" & dat$datetime < "2023-11-01", ]
test  <- dat[dat$datetime >= "2023-11-01", ]
cat("Train:", nrow(train), " Val:", nrow(val), " Test:", nrow(test), "\n")


########## helpers ##########

get.metrics <- function(mod, newdata, logy = TRUE) {
  if(logy) {
    pred <- exp(predict(mod, newdata = newdata))
  } else {
    pred <- predict(mod, newdata = newdata)
  }
  actual <- newdata$demand
  rmse <- sqrt(mean((actual - pred)^2))
  mape <- mean(abs((pred - actual)/actual))*100

  if(logy) {
    pi <- exp(predict(mod, newdata = newdata, interval = "prediction", level = 0.95))
  } else {
    pi <- predict(mod, newdata = newdata, interval = "prediction", level = 0.95)
  }
  picp <- mean(actual >= pi[,"lwr"] & actual <= pi[,"upr"])

  s <- summary(mod)
  cat(sprintf("  R2=%.4f  AdjR2=%.4f  sigma=%.2f  p=%d\n", s$r.squared, s$adj.r.squared, s$sigma, length(coef(mod))))
  cat(sprintf("  RMSE=%.0f  MAPE=%.2f%%  PICP=%.3f\n", rmse, mape, picp))
  return(list(pred=pred, rmse=rmse, mape=mape, picp=picp))
}

diag.plots <- function(mod, traindata, tag) {
  par(mfrow = c(2,2))
  plot(mod$fitted, mod$residuals, pch=".", main=paste(tag, "residuals vs fitted"))
  abline(h=0, col="red")
  plot(mod$fitted, abs(mod$residuals), pch=".", main=paste(tag, "|residuals| vs fitted"))
  set.seed(562)
  idx <- sample(1:nrow(traindata), min(5000, nrow(traindata)))
  qqnorm(rstandard(mod)[idx], main=paste(tag, "normal QQ"))
  qqline(rstandard(mod)[idx])
  hist(mod$residuals, breaks=50, main=paste(tag, "residual histogram"),
       col="lightblue", freq=FALSE)
  curve(dnorm(x, mean(mod$residuals), sd(mod$residuals)), add=TRUE, col="red", lwd=2)
  par(mfrow = c(1,1))
}



# ---------- baseline OLS, all 19 weather + covid, raw demand ----------
cat("\n-- baseline --\n")

mod1 <- lm(demand ~ temperature + humidity + apparent_temp +
             precipitation + rain + snowfall + snow_depth +
             cloud_cover + et0 + vpd + wind_speed + wind_direction +
             wind_gusts + soil_temp + shortwave_rad + direct_rad +
             is_day + soil_moisture + direct_rad_inst +
             hour + dow + season + covid_flag, data = train)

m1 <- get.metrics(mod1, test, logy = FALSE)
diag.plots(mod1, train, "baseline")

# fan shape, right skew, QQ tails -> transform needed
v1 <- vif(mod1)
cat("VIF > 10:\n"); print(sort(v1[v1 > 10], decreasing = TRUE))


########## drop the obviously collinear ones ##########
# rain ~ precip (r=0.98), direct_rad_inst ~ direct_rad (r=0.99),
# soil_temp ~ temp (r=0.97), et0 ~ radiation (r=0.93),
# wind_gusts ~ wind (r=0.95), direct_rad ~ shortwave (r=0.95)

cat("\n-- after VIF cleanup --\n")

mod2 <- lm(demand ~ temperature + humidity + apparent_temp +
             precipitation + snowfall + snow_depth +
             cloud_cover + vpd + wind_speed + wind_direction +
             shortwave_rad + is_day + soil_moisture +
             hour + dow + season + covid_flag, data = train)

m2 <- get.metrics(mod2, test, logy = FALSE)
v2 <- vif(mod2)
cat("VIF > 10:\n"); print(sort(v2[v2 > 10], decreasing = TRUE))
# temp/apparent still ~400 but dropping either loses meaningful R2


# ---------- BIC backward selection ----------
cat("\n-- BIC selection --\n")
mod3 <- step(mod2, direction = "backward", k = log(nrow(train)), trace = 0)
cat("BIC dropped:", setdiff(names(coef(mod2)), names(coef(mod3))), "\n")
m3 <- get.metrics(mod3, test, logy = FALSE)



# ---------- Box-Cox / log transform ----------
cat("\n-- Box-Cox, log response --\n")
boxcox(mod3, plotit = TRUE)
# peak slightly negative, log sits at the CI boundary -> log for interpretability

mod4 <- lm(log(demand) ~ temperature + humidity + apparent_temp +
             cloud_cover + vpd + wind_speed + shortwave_rad +
             is_day + soil_moisture + hour + dow + season + covid_flag,
           data = train)

s4 <- summary(mod4)
cat("humidity   p:", round(s4$coefficients["humidity","Pr(>|t|)"], 3), "\n")
cat("is_day     p:", round(s4$coefficients["is_day","Pr(>|t|)"], 3), "\n")
cat("covid_flag p:", round(s4$coefficients["covid_flag","Pr(>|t|)"], 3), "\n")

mod4b <- lm(log(demand) ~ temperature + apparent_temp +
              cloud_cover + vpd + wind_speed + shortwave_rad +
              soil_moisture + hour + dow + season + covid_flag,
            data = train)

m4 <- get.metrics(mod4b, test)
diag.plots(mod4b, train, "log model")
# fan reduced, QQ much better




########## B-spline on apparent_temp + holiday (PRIMARY, day-ahead) ##########
cat("\n-- primary model --\n")

plot(train$apparent_temp, mod4b$residuals, pch=".",
     xlab="apparent temp", ylab="residuals",
     main="residuals vs apparent temp (nonlinear pattern)")
abline(h=0, col="red")

# knots at 30 (heating), 50 (comfort band), 70 (cooling onset)
mod5 <- lm(log(demand) ~ bs(apparent_temp, knots = c(30, 50, 70),
                            Boundary.knots = c(-30, 105)) +
             cloud_cover + vpd + shortwave_rad +
             soil_moisture + hour + dow + season + holiday + covid_flag,
           data = train)

m5 <- get.metrics(mod5, test)
diag.plots(mod5, train, "primary")

v5 <- vif(mod5)
cat("GVIF table:\n"); print(round(v5, 3))

cat("\nprimary coefficients:\n")
print(round(coef(mod5), 6))

plot(test$datetime, test$demand/1e6, type="l", col="black",
     xlab="date", ylab="demand (MW)", main="primary: actual vs predicted")
lines(test$datetime, m5$pred/1e6, col="blue")
legend("topright", c("actual","predicted"), col=c("black","blue"), lwd=1)



# ---------- add lag1 demand ----------
cat("\n-- adding lag1 --\n")

train$lag1 <- c(NA, train$demand[-nrow(train)])
val$lag1   <- c(train$demand[nrow(train)], val$demand[-nrow(val)])
test$lag1  <- c(val$demand[nrow(val)], test$demand[-nrow(test)])

# 24h trailing mean of apparent_temp
train$temp_roll24 <- as.numeric(stats::filter(train$apparent_temp, rep(1/24, 24), sides = 1))
val$temp_roll24   <- as.numeric(stats::filter(val$apparent_temp,   rep(1/24, 24), sides = 1))
test$temp_roll24  <- as.numeric(stats::filter(test$apparent_temp,  rep(1/24, 24), sides = 1))

train.c <- train[complete.cases(train[,c("lag1","temp_roll24")]), ]
val.c   <- val[complete.cases(val[,c("lag1","temp_roll24")]), ]
test.c  <- test[complete.cases(test[,c("lag1","temp_roll24")]), ]

mod6 <- lm(log(demand) ~ bs(apparent_temp, knots = c(30, 50, 70),
                            Boundary.knots = c(-30, 105)) +
             log(lag1) + cloud_cover + vpd + shortwave_rad +
             soil_moisture + hour + dow + season + holiday + covid_flag,
           data = train.c)

m6 <- get.metrics(mod6, test.c)

plot(test.c$datetime, test.c$demand/1e6, type="l", col="black",
     xlab="date", ylab="demand (MW)", main="with lag1: actual vs predicted")
lines(test.c$datetime, m6$pred/1e6, col="blue")
legend("topright", c("actual","predicted"), col=c("black","blue"), lwd=1)


# ---------- hour*season + roll24 ----------
cat("\n-- hour x season interaction --\n")

mod7 <- lm(log(demand) ~ bs(apparent_temp, knots = c(30, 50, 70),
                            Boundary.knots = c(-30, 105)) +
             log(lag1) + temp_roll24 + cloud_cover + vpd + shortwave_rad +
             soil_moisture + hour * season + dow + holiday + covid_flag,
           data = train.c)

m7 <- get.metrics(mod7, test.c)
cat("coefficients:", length(coef(mod7)), "\n")
diag.plots(mod7, train.c, "hour x season")

par(mfrow = c(1,2))
plot(train.c$apparent_temp, mod7$residuals, pch=".",
     xlab="apparent temp", ylab="residuals", main="residuals vs temp")
abline(h=0, col="red")
boxplot(mod7$residuals ~ train.c$hour, main="residuals by hour", xlab="hour")
abline(h=0, col="red")
par(mfrow = c(1,1))

# course method: successive residual scatter
n7 <- length(mod7$residuals)
plot(mod7$residuals[-n7], mod7$residuals[-1], pch=".",
     xlab = expression(hat(epsilon)[i]),
     ylab = expression(hat(epsilon)[i+1]),
     main = "successive residuals")
abline(h=0, col="red"); abline(v=0, col="red")
# positive correlation visible -> motivates lag1 inclusion


# overfit check
pred7.tr <- exp(predict(mod7, newdata = train.c))
rmse7.tr <- sqrt(mean((train.c$demand - pred7.tr)^2))
cat("train RMSE:", round(rmse7.tr,0),
    " test RMSE:", round(m7$rmse,0),
    " ratio:", round(m7$rmse / rmse7.tr, 3), "\n")


# 10-fold CV
cv7 <- cv.glm(train.c,
              glm(log(demand) ~ bs(apparent_temp, knots = c(30,50,70),
                                   Boundary.knots = c(-30,105)) +
                    log(lag1) + temp_roll24 + cloud_cover + vpd + shortwave_rad +
                    soil_moisture + hour*season + dow + holiday + covid_flag,
                  data = train.c),
              K = 10)
cat("CV RMSE (log):", round(sqrt(cv7$delta[1]), 5),
    " vs sigma:", round(summary(mod7)$sigma, 5), "\n")

plot(test.c$datetime, test.c$demand/1e6, type="l", col="black",
     xlab="date", ylab="demand (MW)", main="hour x season: actual vs predicted")
lines(test.c$datetime, m7$pred/1e6, col="blue")
legend("topright", c("actual","predicted"), col=c("black","blue"), lwd=1)




########## group hours -> fewer parameters (ENHANCED, 1h-ahead) ##########
# 24 x 4 = 96 interaction terms is a lot; collapse hours into 5 groups
cat("\n-- enhanced (grouped hours) --\n")

for(d in c("train.c","val.c","test.c")) {
  x <- get(d)
  h <- as.numeric(as.character(x$hour))
  x$hour_grp <- factor(ifelse(h %in% 0:5, "night",
                       ifelse(h %in% 6:8, "morning",
                       ifelse(h %in% 9:16, "daytime",
                       ifelse(h %in% 17:20, "evening", "late_night")))),
                       levels = c("night","morning","daytime","evening","late_night"))
  assign(d, x)
}

mod8 <- lm(log(demand) ~ bs(apparent_temp, knots = c(30, 50, 70),
                            Boundary.knots = c(-30, 105)) +
             log(lag1) + temp_roll24 + cloud_cover + vpd + shortwave_rad +
             soil_moisture + hour_grp * season + dow + holiday + covid_flag,
           data = train.c)

m8 <- get.metrics(mod8, test.c)
cat("coefficients: interaction=", length(coef(mod7)),
    "  grouped=", length(coef(mod8)), "\n")
diag.plots(mod8, train.c, "enhanced")

# leverage + Cook's distance
par(mfrow = c(1,2))
p8 <- length(coef(mod8)) - 1
n8 <- nrow(train.c)
plot(hatvalues(mod8), pch=".", main="leverage")
abline(h = 2*(p8+1)/n8, col="red", lty=2)
cd8 <- cooks.distance(mod8)
plot(cd8, type="h", main="Cook's distance")
par(mfrow = c(1,1))
cat("max Cook's D:", round(max(cd8), 4), "\n")

n8r <- length(mod8$residuals)
plot(mod8$residuals[-n8r], mod8$residuals[-1], pch=".",
     xlab = expression(hat(epsilon)[i]),
     ylab = expression(hat(epsilon)[i+1]),
     main = "successive residuals (enhanced)")

# WLS reweighting for remaining heteroscedasticity
abs.res8 <- abs(mod8$residuals)
var.mod8 <- lm(log(abs.res8^2) ~ mod8$fitted.values)
w8 <- 1 / exp(fitted(var.mod8))
mod8.wls <- lm(log(demand) ~ bs(apparent_temp, knots = c(30,50,70),
                                Boundary.knots = c(-30,105)) +
                 log(lag1) + temp_roll24 + cloud_cover + vpd + shortwave_rad +
                 soil_moisture + hour_grp*season + dow + holiday + covid_flag,
               data = train.c, weights = w8)
m8w <- get.metrics(mod8.wls, test.c)


plot(test.c$datetime, test.c$demand/1e6, type="l", col="black",
     xlab="date", ylab="demand (MW)", main="enhanced: actual vs predicted")
lines(test.c$datetime, m8$pred/1e6, col="blue")
legend("topright", c("actual","predicted"), col=c("black","blue"), lwd=1)


# last 14 days with 95% PI band
last14 <- test.c[test.c$datetime >= max(test.c$datetime) - 14*86400, ]
pred.14 <- exp(predict(mod8, newdata = last14))
pi.14   <- exp(predict(mod8, newdata = last14, interval = "prediction", level = 0.95))

plot(last14$datetime, last14$demand/1e6, type="l", lwd=2, col="black",
     xlab="date", ylab="demand (MW)",
     main="last 2 weeks with 95% prediction interval",
     ylim = range(pi.14)/1e6)
lines(last14$datetime, pred.14/1e6, col="blue", lwd=2)
polygon(c(last14$datetime, rev(last14$datetime)),
        c(pi.14[,"lwr"]/1e6, rev(pi.14[,"upr"]/1e6)),
        col = rgb(0,0,1,0.15), border = NA)
legend("topright", c("actual","predicted","95% PI"),
       col = c("black","blue", rgb(0,0,1,0.3)), lwd = c(2,2,10))

cat("\nenhanced coefficients:\n")
print(round(coef(mod8), 6))



# ---------- recursive forecasting (multi-hour rollout) ----------
cat("\n-- recursive rollout --\n")

test.rec <- test.c
test.rec$pred <- NA

for(i in 1:nrow(test.rec)) {
  test.rec$pred[i] <- exp(predict(mod8, newdata = test.rec[i,]))
  if(i < nrow(test.rec)) {
    gap <- as.numeric(difftime(test.rec$datetime[i+1], test.rec$datetime[i], units = "hours"))
    if(!is.na(gap) && gap <= 1) {
      test.rec$lag1[i+1] <- test.rec$pred[i]
    }
  }
}

pred.std <- m8$pred
rmse.std <- m8$rmse
rmse.rec <- sqrt(mean((test.rec$demand - test.rec$pred)^2))
mape.std <- m8$mape
mape.rec <- mean(abs((test.rec$pred - test.rec$demand) / test.rec$demand)) * 100

cat(sprintf("standard (actual lag1): RMSE=%.0f  MAPE=%.2f%%\n", rmse.std, mape.std))
cat(sprintf("recursive (fed lag1):   RMSE=%.0f  MAPE=%.2f%%\n", rmse.rec, mape.rec))


par(mfrow = c(2,1), mar = c(4,4,2,1))
plot(test.c$datetime, test.c$demand/1e6, type="l", col="black",
     xlab="", ylab="MW", main="standard (actual lag1)")
lines(test.c$datetime, pred.std/1e6, col="blue")
plot(test.rec$datetime, test.rec$demand/1e6, type="l", col="black",
     xlab="date", ylab="MW", main="recursive (cascading lag1)")
lines(test.rec$datetime, test.rec$pred/1e6, col="red")
par(mfrow = c(1,1))




# first 2 weeks: standard vs recursive
first14   <- test.c[test.c$datetime <= min(test.c$datetime) + 14*86400, ]
first14.r <- test.rec[test.rec$datetime <= min(test.rec$datetime) + 14*86400, ]
pred.f14  <- exp(predict(mod8, newdata = first14))

par(mfrow = c(2,1), mar = c(4,4,2,1))
plot(first14$datetime, first14$demand/1e6, type="l", lwd=2, col="black",
     xlab="date", ylab="MW", main="first 2 weeks: standard vs recursive")
lines(first14$datetime, pred.f14/1e6, col="blue", lwd=1.5)
lines(first14.r$datetime, first14.r$pred/1e6, col="red", lwd=1.5)
legend("topright", c("actual","standard","recursive"),
       col = c("black","blue","red"), lwd = c(2,1.5,1.5))

# need last14 recursive too
last14.r <- test.rec[test.rec$datetime >= max(test.rec$datetime) - 14*86400, ]
pred.s14 <- exp(predict(mod8, newdata = last14.r))

plot(last14.r$datetime, last14.r$demand/1e6, type="l", lwd=2, col="black",
     xlab="date", ylab="MW", main="last 2 weeks: standard vs recursive")
lines(last14.r$datetime, pred.s14/1e6, col="blue", lwd=1.5)
lines(last14.r$datetime, last14.r$pred/1e6, col="red", lwd=1.5)
legend("topright", c("actual","standard","recursive"),
       col = c("black","blue","red"), lwd = c(2,1.5,1.5))
par(mfrow = c(1,1))


########## coverage diagnostics ##########
cat("\n-- PICP by season --\n")
pi8 <- exp(predict(mod8, newdata = test.c, interval = "prediction", level = 0.95))
for(s in c("winter","spring","summer","fall")) {
  ii <- test.c$season == s
  if(sum(ii) > 0) {
    cov <- mean(test.c$demand[ii] >= pi8[ii,"lwr"] &
                test.c$demand[ii] <= pi8[ii,"upr"])
    cat(s, "PICP:", round(cov,3), " n=", sum(ii), "\n")
  }
}

hot  <- test.c$apparent_temp > 90
cold <- test.c$apparent_temp < 10
norm <- test.c$apparent_temp >= 30 & test.c$apparent_temp <= 70
if(sum(hot)  > 0) cat("hot PICP:",  round(mean(test.c$demand[hot]  >= pi8[hot,"lwr"]  & test.c$demand[hot]  <= pi8[hot,"upr"]),  3), "\n")
if(sum(cold) > 0) cat("cold PICP:", round(mean(test.c$demand[cold] >= pi8[cold,"lwr"] & test.c$demand[cold] <= pi8[cold,"upr"]), 3), "\n")
cat("normal PICP:", round(mean(test.c$demand[norm] >= pi8[norm,"lwr"] & test.c$demand[norm] <= pi8[norm,"upr"]), 3), "\n")


# daily average % error on test
pct.err <- (pred.std - test.c$demand)/test.c$demand * 100
daily <- tapply(pct.err, as.Date(test.c$datetime), mean)
plot(as.Date(names(daily)), daily, type="l", col="blue",
     xlab="date", ylab="daily % error", main="daily average error on test set")
abline(h=0, col="red", lwd=2)
abline(h=c(-5,5), col="grey", lty=2)



# ---------- results summary ----------
cat("\n\n==== APPROACH 1 RESULTS (COVID KEPT + FLAG) ====\n\n")
cat(sprintf("%-40s %8s %10s %8s %6s %4s\n", "model", "AdjR2", "TestRMSE", "MAPE%", "PICP", "p"))
cat(strrep("-", 78), "\n")

info <- list(
  list("baseline (19 weather + covid, raw)",  mod1,  m1,  FALSE),
  list("VIF cleanup (13 weather + covid)",    mod2,  m2,  FALSE),
  list("BIC selection",                       mod3,  m3,  FALSE),
  list("log(demand), drop insig",             mod4b, m4,  TRUE),
  list("B-spline + holiday [PRIMARY]",        mod5,  m5,  TRUE),
  list("+ lag1",                              mod6,  m6,  TRUE),
  list("+ hour*season + roll24",              mod7,  m7,  TRUE),
  list("grouped hours [ENHANCED]",            mod8,  m8,  TRUE)
)

for(x in info) {
  cat(sprintf("%-40s %8.4f %10.0f %8.2f %6.3f %4d\n",
              x[[1]], summary(x[[2]])$adj.r.squared, x[[3]]$rmse,
              x[[3]]$mape, x[[3]]$picp, length(coef(x[[2]]))))
}

cat(sprintf("\n%-40s %8s %10.0f %8.2f %6s %4s\n",
            "recursive (cascading lag1)", "--", rmse.rec, mape.rec, "--", "--"))

cat("\nprimary  = day-ahead (no lag needed)\n")
cat("enhanced = 1-hour-ahead (uses previous demand)\n")
cat("recursive = multi-hour extension (predicted demand feeds back)\n")
