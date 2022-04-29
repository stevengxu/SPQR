#' @title summary method for class \code{SPQR}
#' @description summarizes the output produced by \code{SPQR()} and structures them in a more organized way to be examined by the user.
#'
#' @method summary SPQR
#' @name summary.SPQR
#'
#' @param object An object of class \code{SPQR}.
#' @param ... Other arguments.
#'
#' @return An object of class \code{summary.SPQR}. A list containing summary information
#' of the fitted model.
#' \item{method}{The estimation method}
#' \item{time}{The elapsed time}
#' \item{prior}{If \code{method = "MAP"} or \code{method = "MCMC"}, the hyperprior model for the variance hyperparameters}
#' \item{model}{If \code{method = "MLE"} or \code{method = "MAP"}, the fitted \code{torch} model. If \code{method = "MCMC"}, the posterior samples of neural network parameters}
#' \item{loss}{If \code{method = "MLE"} or \code{method = "MAP"}, the train and validation loss}
#' \item{optim.info}{If \code{method = "MLE"} or \code{method = "MAP"}, configuration information of the Adam routine}
#' \item{elpd}{If \code{method = "MCMC"}, the expected log-predictive density}
#' \item{diagnostics}{If \code{method = "MCMC"}, diagnostic information of the MCMC chain}
#'
#' @examples
#' \donttest{
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' control <- list(iter = 200, warmup = 150, thin = 1)
#' fit <- SPQR(X = X, Y = Y, method = "MCMC", control = control,
#'             normalize = TRUE, verbose = FALSE)
#'
#' ## summarize output
#' summary(fit)
#' }
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
