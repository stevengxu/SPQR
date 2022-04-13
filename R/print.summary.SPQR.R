#' @method print summary.SPQR
#'
#' @name summary.SPQR
#'
#' @param x an object of class \code{SPQR}
#' @param showMode whether to print the NN structure
#' @param ... other arguments
#'
#' @export
print.summary.SPQR <- function(x, showModel = FALSE, ...) {
  s <- x
  method <- s$method
  if (method != "MLE")
    cat("\nSPQR fitted using ", method, " approach with ", s$prior, " prior", sep="")
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

    bess <- s$diagnostics$bulk.ess
    tess <- s$diagnostics$tail.ess
    ndiv <- s$diagnostics$ndiv
    loo <- s$elpd$loo
    waic <- s$elpd$waic
    accept.ratio <- s$diagnostics$accept.ratio
    delta <- s$diagnostics$delta

    cat("\nMCMC diagnostics:\n",
        "  bulk.ESS = ", bess, ",  tail.ESS = ", tess, "\n", sep="")
    cat("  Final acceptance ratio is ", sprintf("%.2f", accept.ratio), " and target is ", delta, "\n", sep="")
    if (s$diagnostics$ndiv > 0)
      cat("  There were ", paste0(ndiv, " divergent transitions after warmup"), "\n", sep="")

    cat("\nExpected log pointwise predictive density (elpd) estimates:\n",
        "  elpd.LOO = ", loo, ",  elpd.WAIC = ", waic, "\n", sep="")
  } else {
    tr <- s$loss$train
    va <- s$loss$validation
    cat("\nLoss:\n",
        "  train = ", tr, ",  validation = ", va, "\n", sep="")
  }
  cat("\nElapsed time: ", paste0(sprintf("%.2f", s$time), " minutes"), "\n", sep = "")
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
