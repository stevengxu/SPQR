#' @title summaary method for SPQR
#' @description summarizes the output produced by \code{SPQR()} and structures them in a more organized way to be examined by the user.
#'
#' @method summary SPQR
#'
#' @param object An object of class \code{"SPQR"}.
#' @param ... Other arguments.
#'
#' @return An object of class \code{"summary.SPQR"}.
#'
#' @export
summary.SPQR <- function(object, ...) {

  method <- object$method
  out <- list(method=method, time=object$time)
  if (method != "MLE") out$prior <- object$config$prior
  out$model <- list(n.inputs=ncol(object$X),
                    n.knots=object$config$n.knots,
                    n.hidden=object$config$n.hidden,
                    activation=object$config$activation)
  if (method == "MCMC") {
    ll.mat <- object$chain.info$loglik
    suppressWarnings(waic <- loo::waic(ll.mat)$estimates[1]) # Calculate WAIC
    reff <- loo::relative_eff(exp(ll.mat), chain_id=rep(1,nrow(ll.mat)))
    suppressWarnings(loo <- loo::loo(ll.mat, r_eff=reff)$estimates[1]) # Calculate LOOIC
    ndiv <- sum(object$chain.info$divergent)
    out$elpd <- list(loo=loo, waic=waic)
    accept.ratio <- mean(object$chain.info$accept.ratio)
    delta <- object$chain.info$delta
    out$diagnostics <- list(ndiv=ndiv, accept.ratio=accept.ratio, delta=delta)
  } else {
    out$loss <- object$loss
    out$optim.info <- list(lr=object$control$lr,
                           batch.size=object$control$batch.size)
  }
  class(out) <- "summary.SPQR"
  return(out)
}
