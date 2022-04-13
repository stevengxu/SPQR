library(SPQR)

# RV for the simulation design
rYgivenX <- function(n,p){
  X     <- matrix(runif(n*p),n,p)
  p     <- 1/(1+exp(-1+2*(X[,1]>0.5)))
  Z     <- rbinom(n,1,p)
  Y     <- Z*rlnorm(n,0,0.75) + (1-Z)*rnorm(n,2,0.5)
  out   <- list(Y=Y,X=X)
  return(out)}

# PDF for the simulation design
dYgivenX <- function(y,X){
  p     <- 1/(1+exp(-1+2*(X[1]>0.5)))
  out   <- p*dlnorm(y,0,0.75) + (1-p)*dnorm(y,2,0.5)
  return(out)}

# QF for the simulation design
qYgivenX <- function(tau,X){
  m   <- 1/(1+exp(-1+5*X[1]*X[2]))
  out <- qbeta(tau,10*m,10*(1-m))
  return(out)}

# Simulation
# Strong (X1,X2) interaction; X3 has no effect

# RV for the simulation design
rYgivenX <- function(n,p){
  X     <- matrix(runif(n*p),n,p)
  m     <- 1/(1+exp(-1+5*X[,1]*X[,2]))
  Y     <- rbeta(n,10*m,10*(1-m))
  out   <- list(Y=Y,X=X)
  return(out)}

# PDF for the simulation design
dYgivenX <- function(y,X){
  m   <- 1/(1+exp(-1+5*X[1]*X[2]))
  out <- dbeta(y,10*m,10*(1-m))
  return(out)}

# QF for the simulation design
qYgivenX <- function(tau,X){
  m   <- 1/(1+exp(-1+5*X[1]*X[2]))
  out <- qbeta(tau,10*m,10*(1-m))
  return(out)}

set.seed(919)
n   <- 2000   # Number of observations
p   <- 3      # Number of covariates
dat <- rYgivenX(n,p)
X   <- dat$X
Y   <- dat$Y

# not all parameters need to be specified, see utils.R for default values
mle.control <- list(
  lr = exp(-4), # learning rate
  dropout = c(0,0), # a tuple with first element indicating dropout prob for input layer
                    # and second element for common dropout prob for hidden layers
  batchnorm = F,
  batch.size = 256,
  epochs = 500,
  early.stopping.epochs = 10, # early stopping number
  save.path = file.path(getwd(),"SPQR_model"), # path to save the best torch model during validation
  save.name = "SPQR.model.pt" # file to save the ...
)

# using adam to obtain mle estimate
mle.fit <- SPQR(X=X, Y=Y, n.knots=8, n.hidden=10, activation="tanh",
                method="MLE", control=mle.control, use.GPU=F)
# obtain summary of fitted model
mle.summary <- summary(mle.fit)
mle.summary # directly calling mle.fit will give same result
# print summary info and show the NN structure
print(mle.summary, showModel=T)
# Do the previous two-steps together
print(mle.fit, showModel=T)

# Calculate K-fold CV error
folds <- createFolds.SPQR(Y, nfold=5)
cv.out <- cv.SPQR(X=X, Y=Y, folds=folds, n.knots=12, n.hidden=10, activation="tanh",
                  method="MLE", control=mle.control, use.GPU=T)

# alternatively the user can use pre-computed folds
# this can be wrapped in a gridsearch skeleton for parameter selection purpose
if (FALSE) {
  folds <- SPQR.createFolds(Y, nfold=5)
  lr.grid <- exp(-5:-1)
  result <- do.call('rbind',lapply(lr.grid, FUN=function(lr) {
    adam.params$lr <- lr
    cv.out <- cv.SPQR(method="MLE", params=adam.params, X=X, Y=Y, n.hidden=n.hidden,
                      n.knots=n.knots, activation=activation, folds=folds)
    c(lr, cv.out$cve)
  }))
  colnames(result) <- c("lr","cve")
}

# using adam to obtain map estimate
# MAP can be more sensitive to learning rate
map.control <- mle.control
map.control$lr <- exp(-5)
map.fit <- SPQR(X=X, Y=Y, n.knots=8, n.hidden=10, activation="tanh",
                method="MAP", prior="ARD", control=map.control, use.GPU=F)
# obtain summary of fitted model
map.summary <- summary(map.fit)
# print summary info
print(map.summary, showModel=T)
# Do the previous two-steps together
print(map.fit, showModel=T)

mcmc.control <- list(
  iter = 1000, # total number of iterations
  warmup = 250, # warmup iters for stepsize and mass matrix adaptation
  thin = 5 # period for saving posterior samples
)

# using mcmc to obtain bayes estimate
bayes.fit <- SPQR(X=X, Y=Y, n.knots=8, n.hidden=10, activation="tanh",
                  method="MCMC", prior="GSM", control=mcmc.control)
# obtain summary of fitted model
bayes.summary <- summary(bayes.fit)
# print summary info
print(bayes.summary, showModel=T)
# Do the previous two-steps together
print(bayes.fit, showModel=T)

# prediction based on fitted model
X_test <- X[1:9,]
# PDF
# not specifying `Y` means we want to full curve
# default is seq(0,1,length.out=501)
pdf.mle <- predict(mle.fit, X=X_test, type="PDF")
pdf.map <- predict(map.fit, X=X_test, type="PDF")
pdf.bayes <- predict(bayes.fit, X=X_test, type="PDF")
# the output is a named array
names(dimnames(pdf.bayes))

yyy <- seq(0,1,length.out=501)
par(mfrow=c(3,3))
for(i in 1:9){
  pdf0 <- dYgivenX(yyy,X_test[i,]) # True
  pdf1 <- pdf.mle[i,]     # MLE
  pdf2 <- pdf.map[i,]     # MAP
  pdf3 <- pdf.bayes[i,]   # Bayes
  plot(yyy,pdf0,ylim=c(0,1.5*max(pdf0)),type="l",
       xlab="y",ylab="PDF",main=paste("Observation",i))
  lines(yyy,pdf1,col=2)
  lines(yyy,pdf2,col=3)
  lines(yyy,pdf3,col=4)
  if(i==1){
    legend("topright",c("True","MLE","MAP","Bayes"),lty=1,col=1:4,bty="n")
  }
}

# the built-in plotting function plots PDF/CDF/QF curve for a single X
plot(mle.fit, X=X_test[1,], type="PDF")
plot(map.fit, X=X_test[1,], type="PDF")
plot(bayes.fit, X=X_test[1,], type="PDF")

# pdf points estimates and credible intervals
pdf.bayes <- predict(bayes.fit, X_test, type="PDF", ci.level=0.95)
names(dimnames(pdf.bayes))

par(mfrow=c(3,3))
for(i in 1:9){
  pdf0 <- dYgivenX(yyy,X_test[i,]) # True
  pdf1.lb <- pdf.bayes[i,,"lower.bound"]
  pdf1 <- pdf.bayes[i,,"mean"]
  pdf1.ub <- pdf.bayes[i,,"upper.bound"]
  plot(yyy,pdf0,ylim=c(0,1.5*max(pdf0)),type="l",
       xlab="y",ylab="PDF",main=paste("Observation",i))
  lines(yyy,pdf1,col=2)
  lines(yyy,pdf1.lb,col=2,lty=2)
  lines(yyy,pdf1.ub,col=2,lty=2)
  if(i==1){
    legend("topright",c("True","Bayes"),lty=1,col=1:2,bty="n")
  }
}

# using built-in plotting function
plot(bayes.fit, X=X_test[1,], type="PDF", ci.level=0.95)
# instead of credible bands, plot all posterior samples
plot(bayes.fit, X=X_test[1,], type="PDF", getAll=TRUE)

# quantile function
tau <- seq(0.05,0.95,0.05)
qf.mle <- predict(mle.fit, X_test, type="QF", tau=tau)
qf.map <- predict(map.fit, X_test, type="QF", tau=tau)
qf.bayes <- predict(bayes.fit, X_test, type="QF", tau=tau)
names(dimnames(qf.bayes))

par(mfrow=c(3,3))
for(i in 1:9){
  qf0 <- qYgivenX(tau,X_test[i,]) # True
  qf1 <- qf.mle[i,]     # MLE
  qf2 <- qf.map[i,]     # MAP
  qf3 <- qf.bayes[i,]   # Bayes
  plot(tau,qf0,ylim=c(0,1.5*max(qf0)),type="l",
       xlab="tau",ylab="Quantile",main=paste("Observation",i))
  lines(tau,qf1,col=2)
  lines(tau,qf2,col=3)
  lines(tau,qf3,col=4)
  if(i==1){
    legend("topright",c("True","MLE","MAP","Bayes"),lty=1,col=1:4,bty="n")
  }
}

# qf points estimates and credible intervals
qf.bayes <- predict.SPQR(bayes.fit, X_test, type="QF", tau=tau, ci.level=0.95)
par(mfrow=c(3,3))
for(i in 1:9){
  qf0 <- qYgivenX(tau,X_test[i,]) # True
  qf1.lb <- qf.bayes[i,,"lower.bound"]
  qf1 <- qf.bayes[i,,"mean"]
  qf1.ub <- qf.bayes[i,,"upper.bound"]
  plot(tau,qf0,ylim=c(0,1.5*max(qf0)),type="l",
       xlab="tau",ylab="Quantile",main=paste("Observation",i))
  lines(tau,qf1,col=2)
  lines(tau,qf1.lb,col=2,lty=2)
  lines(tau,qf1.ub,col=2,lty=2)
  if(i==1){
    legend("topright",c("True","Bayes"),lty=1,col=1:2,bty="n")
  }
}

# using built-in plotting function
plot(bayes.fit, X=X_test[1,], type="QF", ci.level=0.95)
# instead of credible bands, plot all posterior samples
plot(bayes.fit, X=X_test[1,], type="QF", getAll=TRUE)

# Goodness-of-fit based on inverse transform method
qqCheck(mle.fit)
qqCheck(map.fit)
qqCheck(bayes.fit)
# same but with credible bands
qqCheck(bayes.fit, ci.level=0.95)
# with all posterior samples
qqCheck(bayes.fit, getAll=TRUE)

# Sensitivity analysis
tau <- c(0.25,0.5,0.75)
# True generating function
pred.fun <- function(X, tau) {
  out <- matrix(nrow=nrow(X), ncol=length(tau))
  for (i in 1:nrow(X)) {
    out[i,] <- qYgivenX(tau, X[i,])
  }
  return(out)
}

# Main effect for x2, x3, x4
par(mfrow=c(3,3))
for (j in 1:3) {
  ale.mle <- QALE(mle.fit, var.index=j, tau=tau)
  ale.map <- QALE(map.fit, var.index=j, tau=tau)
  ale.bayes <- QALE(bayes.fit, var.index=j, tau=tau)
  ale.ans <- QALE(list(X=X), var.index=j, tau=tau, pred.fun=pred.fun)

  for (i in 1:length(tau)) {
    plot(ale.ans$x, ale.ans$ALE[,i], type="l", xlab=parse(text=paste0("X[",j,"]")), ylab="ALE", col=1)
    lines(ale.mle$x, ale.mle$ALE[,i], col=2)
    lines(ale.map$x, ale.map$ALE[,i], col=3)
    lines(ale.bayes$x, ale.bayes$ALE[,i], col=4)
    if(i==1){
      legend("topright",c("True","MLE","MAP","Bayes"),lty=1,col=1:4,bty="n")
    }
  }
}

# using built-in plotQALE
plotQALE(bayes.fit, var.index=1, tau=seq(0.1,0.9,0.1))
# with credible bands
# SLOW!!
plotQALE(bayes.fit, var.index=1, tau=seq(0.1,0.9,0.1), ci.level=0.95)
# with all posterior samples
# SLOW!!
plotQALE(bayes.fit, var.index=1, tau=seq(0.1,0.9,0.1), getAll=TRUE)


# Interaction effect for (X1,X2)
par(mfrow=c(2,2))
ale.mle <- QALE(mle.fit, var.index=c(1,2), tau=tau)
ale.map <- QALE(map.fit, var.index=c(1,2), tau=tau)
ale.bayes <- QALE(bayes.fit, var.index=c(1,2), tau=tau)
ale.ans <- QALE(list(X=X), var.index=c(1,2), tau=tau, pred.fun=pred.fun)

image(ale.ans$x[[1]], ale.ans$x[[2]], ale.ans$ALE[,,1],xlab = parse(text="X[2]"),ylab = parse(text="X[3]"),main = "True")
contour(ale.ans$x[[1]], ale.ans$x[[2]], ale.ans$ALE[,,1], add=TRUE, drawlabels=TRUE)
image(ale.mle$x[[1]], ale.mle$x[[2]], ale.mle$ALE[,,1], xlab = parse(text="X[2]"),ylab = parse(text="X[3]"),main = "MLE")
contour(ale.mle$x[[1]], ale.mle$x[[2]], ale.mle$ALE[,,1], add=TRUE, drawlabels=TRUE)
image(ale.map$x[[1]], ale.map$x[[2]], ale.map$ALE[,,1], xlab = parse(text="X[2]"),ylab = parse(text="X[3]"),main = "MAP")
contour(ale.map$x[[1]], ale.map$x[[2]], ale.map$ALE[,,1], add=TRUE, drawlabels=TRUE)
image(ale.bayes$x[[1]], ale.bayes$x[[2]], ale.bayes$ALE[,,1], xlab = parse(text="X[2]"),ylab = parse(text="X[3]"),main = "Bayes")
contour(ale.bayes$x[[1]], ale.bayes$x[[2]], ale.bayes$ALE[,,1], add=TRUE, drawlabels=TRUE)

par(mfrow=c(1,1))

# using built-in plotQALE
plotQALE(bayes.fit, var.index=c(1,2), tau=seq(0.1,0.9,0.1))
plotQALE(bayes.fit, var.index=c(2,3), tau=seq(0.1,0.9,0.1))

# Variable importance across tau
# not specifing var.index means to consider all features
plotQVI(bayes.fit, tau=seq(0.1,0.9,0.1))
# Variable importance across tau with credible intervals
# VERY SLOW!!
plotQVI(bayes.fit, tau=seq(0.1,0.9,0.1), ci.level=0.95)




# Electric grid data

library(caret)
library(tidyr)
library(dplyr)
data("AUDem", package = "qgam")
meanDem <- AUDem$meanDem



meanDem <- meanDem %>%
  mutate(value=1) %>%
  spread(dow, value, fill=0)

y <- meanDem$dem
X <- as.matrix(meanDem %>% select(-c(date,dem,dem48)))
X.max <- colMaxs(X)
X.min <- colMins(X)
y.max <- max(y)
y.min <- min(y)
X <- t((t(X) - X.min) / (X.max - X.min))
y <- (y - y.min ) / (y.max - y.min)

mcmc.params <- list(
  sampler = "NUTS", # "HMC" also available
  prior = "ARD", # "GP" and "GSM" also available
  n.hidden = 30,
  n.knots = 12,
  activation = "tanh",
  control = list(adapt_delta = 0.9), # HMC control parameters
  iter = 500, # total number of iterations
  warmup = 250, # warmup iters for stepsize and mass matrix adaptation
  thin = 1 # period for saving posterior samples
)

# using mcmc to obtain bayes estimate
bayes.fit <- spqr.train(method="Bayes", params=mcmc.params, X=X, y=y, verbose=2)


autoplot(bayes.fit, type="ALE", var.index=2, tau=seq(0.1,0.9,0.1))



# Simulation 1

set.seed(123)
X <- matrix(runif(10*3),10,3)
m <- 1/(1+exp(-1+5*X[,1]*X[,2]))
Y <- seq(0,1,0.01)
d <- sapply(m,function(u){dbeta(Y,10*u,10*(1-u))})
matplot(Y,d,type="l")

df <- data.frame(y=rep(Y,nrow(X)),d=c(d),obs=rep(1:nrow(X),each=length(Y)))
p1 <- ggplot(data=df,aes(x=y,y=d))+
  geom_line(aes(group=obs,color=factor(obs)))+
  theme_bw()+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="none",
        axis.title = element_text(size = 15),
        plot.title = element_text(hjust = 0.5,size = 18),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(size = 12)) +
  labs(x="Y",y=TeX("$f(Y|X)$")) +
  ggtitle("Conditional Density Functions")

X <- cbind(seq(0,1,0.01),0.1)
m <- 1/(1+exp(-1+5*X[,1]*X[,2]))
tau <- seq(0.05,0.95,0.05)
q <- sapply(m,function(u){qbeta(tau,10*u,10*(1-u))})
matplot(X[,1],t(q),type="l",col=1)

df <- data.frame(X=X[,1],q=c(t(q)),tau=rep(tau,each=nrow(X)))
p2 <- ggplot(data=df,aes(x=X,y=q))+
  geom_line(aes(group=tau))+
  theme_bw()+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="none",
        axis.title = element_text(size = 15),
        plot.title = element_text(hjust = 0.5,size = 18),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(size = 12)) +
  labs(x=TeX("$X_1$"),y=TeX("$Q(\\tau | X_1 , X_2 = 0.1)$")) +
  ggtitle(TeX("Quantile curves at $X_2=0.1$"))

cairo_pdf(filename="SPQR_model/sim1dqf.pdf",width = 12, height = 4)

grid.arrange(p1,p2,p3,nrow=1)


# Plot quantile slice

xx <- pracma::meshgrid(seq(0,1,0.01),seq(0,1,0.01))
xxx <- cbind(c(xx$X),c(xx$Y),0.5)
qq <- predict.SPQR(bayes.fit, xxx, yyy, type="qf", tau=c(0.1,0.5,0.9), ci.level=NULL)


fig <- plot_ly(x = xx$X, y = xx$Y, showscale = FALSE)
m <- 1/(1+exp(-1+5*xx$X*xx$Y))
par(mfrow=c(3,3))
for (i in 1:3) {
  q <- qbeta(tau[i],10*m,10*(1-m))
  image(x=seq(0,1,0.01),y=seq(0,1,0.01),z=matrix(q,nrow=101))
  image(x=seq(0,1,0.01),y=seq(0,1,0.01),z=matrix(qq[,i],nrow=101))
}

q <- sapply(tau,function(tau){
  qbeta(tau, 10*m, 10*(1-m))
})


df <- data.frame(Q=c(q,qq),X1=c(xx$X),X2=c(xx$Y),D=rep(c("True","Estimated"),each=length(q)),tau=rep(tau,each=nrow(q)))
ggplot(data=df, aes(x=X1,y=X2)) +
  geom_raster(aes(fill=Q)) +
  geom_contour(aes(z=Q),colour="black") +
  scale_fill_gradientn(colors=rev(brewer.pal(10,"Spectral"))) +
  facet_grid(D~tau) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(0, "lines"),
        axis.title = element_text(size = 15, face="bold"),
        plot.title = element_text(hjust = 0.5,size = 18),
        axis.text.y = element_text(size = 12),
        axis.text.x = element_text(size = 12))
