test_that("SPQR", {
  library(SPQR)
  n <- 5
  X <- runif(n)
  Y <- runif(n,1,2)
  ## Y not between 0 and 1
  expect_error(SPQR(X=X,Y=Y,method="MCMC"),"`Y` must be between 0 and 1")

  Y <- runif(n)
  X[1] <- NA
  ## X has missing values
  expect_error(SPQR(X=X,Y=Y,method="MCMC"),"`X` cannot have missing values")

  X[1] <- "1"
  ## X has non-numeric values
  expect_error(SPQR(X=X,Y=Y,method="MCMC"),"`X` cannot have non-numeric values")
})
