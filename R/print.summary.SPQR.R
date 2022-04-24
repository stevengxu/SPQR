#' @title print method for summary.SPQR
#' @description Print the output produced by summary.SPQR().
#' @name print.summary.SPQR
#'
#' @method print summary.SPQR
#'
#' @param x An object of class \code{summary.SPQR}
#' @param showModel If \code{TRUE}, prints the detailed NN architecture by layer.
#' @param ... Other arguments.
#'
#' @examples
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' control <- list(iter = 300, warmup = 200, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control, normalize = TRUE)
#'
#' ## summarize output
#' summary(fit)
#'
#' @export
print.summary.SPQR <- function(x, showModel = FALSE, ...) {

  method <- x$method
  if (method != "MLE")
    cat("\nSPQR fitted using ", method, " approach with ", x$prior, " prior", sep="")
  else
    cat("\nSPQR fitted using ", method, " approach", sep="")
  cat("\U0001f680\n")

  if (method != "MCMC") {
    lr <- x$optim.info$lr
    batch.size <- x$optim.info$batch.size
    cat("\nLearning rate: ", lr, sep="")
    cat("\nBatch size: ", batch.size, "\n", sep="")
  }



  if (showModel) {
    cat("\nModel specification:\n")
    cat("  ")
    .printNNmat(x$model)
  }

  if (method == "MCMC") {

    ndiv <- x$diagnostics$ndiv
    loo <- x$elpd$loo
    waic <- x$elpd$waic
    accept.ratio <- x$diagnostics$accept.ratio
    delta <- x$diagnostics$delta

    cat("\nMCMC diagnostics:\n",
        "  Final acceptance ratio is ", sprintf("%.2f", accept.ratio), " and target is ", delta, "\n", sep="")
    if (x$diagnostics$ndiv > 0)
      cat("  There were ", paste0(ndiv, " divergent transitions after warmup"), "\n", sep="")

    cat("\nExpected log pointwise predictive density (elpd) estimates:\n",
        "  elpd.LOO = ", loo, ",  elpd.WAIC = ", waic, "\n", sep="")
  } else {
    tr <- x$loss$train
    va <- x$loss$validation
    cat("\nLoss:\n",
        "  train = ", tr, ",  validation = ", va, "\n", sep="")
  }
  cat("\nElapsed time: ", paste0(sprintf("%.2f", x$time), " minutes"), "\n", sep = "")
}


.printNNmat <- function(model) {
  n.layers <- length(model$n.hidden) + 1
  nodes <- c(model$n.inputs, model$n.hidden, model$n.knots)
  mat <- array("", dim=c(n.layers,3),
               dimnames=list(" "=rep("",n.layers),"Layers"=c("Input","Output","Activation")))
  for (l in 1:n.layers) {
    activation <- if (l < n.layers) model$activation else "softmax"
    mat[l,] <- c(nodes[l],nodes[l+1],activation)
  }
  print.default(mat, quote = FALSE, right = TRUE)
}
