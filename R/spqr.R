#' @title Fitting SPQR models
#' @description
#' Main function of the package. Fits SPQR using the maximum likelihood estimation (MLE), maximum \emph{a posterior} (MAP) or
#' Markov chain Monte Carlo (MCMC) method. Returns an object of S3 class `SPQR`.
#'
#' @docType package
#' @useDynLib SPQR
#'
#' @name SPQR
#' @param X the covariate matrix
#' @param Y the response vector
#' @param n.knots number of spline basis to use
#' @param n.hidden number of hidden neurons
#' @param activation hidden layer activation
#' @param method type of estimator to use
#' @param prior prior model for variance hyperparameters
#' @param hyperpar hyperparameter values for the variance hyperpriors
#' @param control a list of named parameters for finer control of the training process
#' @param normalize whether to normalize the response and covariates to unit interval
#' @param verbose whether to print progress during training
#' @param seed random number generation seed
#'
#'
#'
#' @export
SPQR <-
  function(X, Y, n.knots = 12, n.hidden = 10, activation = c("tanh","relu","sigmoid"),
           method=c("MLE","MAP","MCMC"), prior=c("GP","ARD","GSM"), hyperpar=list(),
           control=list(), normalize = FALSE, verbose = TRUE, seed = NULL, ...)
{

  activation <- match.arg(activation)
  method <- match.arg(method)
  prior <- match.arg(prior)

  if (n.knots < 5)
    stop("Very small `n.knots` can lead to severe underfitting, We recommend setting it to at least 5.")

  if (is.null(n <- nrow(X))) dim(X) <- c(length(X),1) # 1D matrix case
  if (n == 0) stop("`X` is empty")
  if (!is.matrix(X)) X <- as.matrix(X) # data.frame case

  ny <- NCOL(Y)
  if (is.matrix(Y) && ny == 1) Y <- drop(Y) # treat 1D matrix as vector
  if (NROW(Y) != n) stop("incompatible dimensions")

  # normalize all covariates
  if (normalize) {
    Y.range <- range(Y)
    Y <- (Y - Y.range[1])/diff(Y.range)
    X.range <- apply(X,2,range)
    X <- apply(X,2,function(x){
      (x - min(x)) / (max(x) - min(x))
    })
  }
  if (min(Y)<0 || max(Y)>1) stop("values of `Y` should be between 0 and 1")

  control <- .check.control(control, method, ...)
  hyperpar <- .update.hyperpar(hyperpar)
  if (method == "MCMC") {
    out <- SPQR.MCMC(X=X, Y=Y, n.knots=n.knots, n.hidden=n.hidden,
                     activation=activation, prior=prior, hyperpar=hyperpar,
                     control=control, verbose=verbose, seed=seed)
    out$method <- method
  } else {
    out <- SPQR.ADAM(X=X, Y=Y, n.knots=n.knots, n.hidden=n.hidden,
                     activation=activation, method=method, prior=prior,
                     hyperpar=hyperpar, control=control, verbose=verbose,
                     seed=seed)
  }
  if (normalize) {
    out$normalize$X <- X.range
    out$normalize$Y <- Y.range
  }
  class(out) <- "SPQR"
  invisible(out)
}


.check.control <- function(control, method, ...) {
  if (!identical(class(control), "list"))
    stop("`control` should be a list")

  # merge parameters from the control and the dots-expansion
  dot_control <- list(...)
  if (length(intersect(names(control),names(dot_control))) > 0)
    stop("Same parameters in `control` and in the call are not allowed. Please check your `control` list.")
  control <- c(control, dot_control)

  name_freqs <- table(names(control))
  multi_names <- names(name_freqs[name_freqs > 1])
  if (length(multi_names) > 0) {
    warning("The following parameters were provided multiple times:\n\t",
            paste(multi_names, collapse = ', '), "\n  Only the last value for each of them will be used.\n")
    for (n in multi_names) {
      del_idx <- which(n == names(control))
      del_idx <- del_idx[-length(del_idx)]
      control[[del_idx]] <- NULL
    }
  }

  # check method specific parameters
  if (method == "MCMC") {
    control$algorithm <- match.arg(control$algorithm, c("NUTS","HMC"))
    control$metric <- match.arg(control$metric, c("diag","unit","dense"))
  }
  control <- .update.control(control, method)
  return(control)
}

.update.hyperpar <- function(hyperpar) {
  default <- list(
    a_sigma = 0.001,
    b_sigma = 0.001,
    a_lambda = 0.5,
    b_lambda = 0.5
  )
  if (length(hyperpar) > 0) {
    for (i in names(hyperpar))
      default[[i]] <- hyperpar[[i]]
  }
  invisible(default)
}

.update.control <- function(control, method) {
  if (method == "MCMC") {
    default <- list(
      algorithm = "NUTS",
      iter = 1000,
      warmup = 500,
      thin = 1,
      stepsize = NULL,
      delta = 0.9,
      metric = "diag",
      max.treedepth = 6,
      int.time = 1,
      #################
      gamma = 0.05,
      kappa = 0.75,
      t0 = 10,
      init.buffer = 75,
      term.buffer = 50,
      base.window = 25
    )
  } else {
    default <- list(
      lr = 0.01,
      dropout = c(0,0),
      batchnorm = FALSE,
      epochs = 200,
      batch.size = 128,
      valid.pct = 0.2,
      early.stopping.epochs = 10,
      print.every.epochs = 10,
      save.path = file.path(getwd(),"SPQR_model"),
      save.name = "SPQR.model.pt"
    )
  }
  if (length(control) > 0) {
    for (i in names(control))
      default[[i]] <- control[[i]]
  }
  invisible(default)
}
