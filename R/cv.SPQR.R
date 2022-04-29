#' @title cross-validation for SPQR estimator
#' @description
#' Fits SPQR using either MLE or MAP method and computes K-fold cross-validation error.
#' @name cv.SPQR
#'
#' @param folds A list of CV folds, possibly that generated from \code{\link[=createFolds.SPQR]{createFolds.SPQR()}}.
#' @inheritParams SPQR
#'
#' @return
#' \item{control}{the list of all control parameters.}
#' \item{cve}{the cross-validation error.}
#' \item{folds}{the CV folds.}
#'
#' @seealso \code{\link[=createFolds.SPQR]{createFolds.SPQR()}}
#' @importFrom torch `%>%` torch_tensor
#'
#' @examples
#' \donttest{
#' set.seed(919)
#' n <- 200
#' X <- rbinom(n, 1, 0.5)
#' Y <- rnorm(n, X, 0.8)
#' folds <- createFolds.SPQR(Y, nfold = 5)
#' ## compute 5-fold CV error
#' # cv.out <- cv.SPQR(folds=folds, X=X, Y=Y, method="MLE",
#' #                   normalize = TRUE, verbose = FALSE)
#' }
#' @export
cv.SPQR <- function(folds, X, Y, n.knots = 10, n.hidden = 10, activation = c("tanh","relu","sigmoid"),
                    method=c("MLE","MAP","MCMC"), prior=c("ARD","GP","GSM"), hyperpar=list(),
                    control=list(), normalize = FALSE, verbose = TRUE, seed = NULL, ...)
{

  activation <- match.arg(activation)
  method <- match.arg(method)
  prior <- match.arg(prior)

  if (n.knots < 5)
    stop("Very small `n.knots` can lead to severe underfitting, We recommend setting it to at least 5.")

  if (is.null(n <- nrow(X))) dim(X) <- c(length(X),1) # 1D matrix case
  n <- nrow(X)
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

  if (!is.list(folds) || length(folds) < 2)
    stop("`folds` must be a list with 2 or more elements that are vectors of indices for each CV-fold")

  out <- cv.SPQR.ADAM(X=X, Y=Y, folds=folds, n.knots=n.knots, n.hidden=n.hidden,
                      activation=activation, method=method, prior=prior,
                      hyperpar=hyperpar, control=control, verbose=verbose,
                      seed=seed)
  invisible(out)
}

cv.SPQR.ADAM <- function(X, Y, folds, n.knots, n.hidden, activation, method, prior,
                         hyperpar, control, verbose = TRUE, seed = NULL) {
  self <- NULL
  use.GPU <- control$use.GPU
  cuda <- torch::cuda_is_available()
  if(use.GPU && !cuda){
    warning('GPU acceleration not available, using CPU')
    use.GPU <- FALSE
  } else if (!use.GPU && cuda){
    message('GPU acceleration is available via `use.GPU=TRUE`')
  }
  device <- if (use.GPU) "cuda" else "cpu"

  K <- length(folds)
  V <- c(ncol(X),n.hidden,n.knots)
  # Define dataset and dataloader
  ds <- torch::dataset(
    initialize = function(indices) {
      self$x <- torch_tensor(X[indices,,drop=FALSE], device=device)
      self$y <- torch_tensor(Y[indices], device=device)
    },

    .getbatch = function(i) {
      list(x=self$x[i,], y=self$y[i], index=i)
    },

    .length = function() {
      self$y$size()[[1]]
    }
  )
  N <- nrow(X)
  # Computing the basis and converting it to a tensor beforehand to save
  # computational time; this is used every iteration for computing loss
  Btotal <- .basis(Y, n.knots)
  # Define custom loss function
  nll.loss = function(indices, basis, coefs) {
    loglik <- basis[indices,]$mul(coefs)$sum(2)$log()$sum()
    return(-loglik)
  }
  cv_losses <- numeric(K)

  for (k in 1:K) {
    valid_indices <- folds[[k]]
    train_indices <- unlist(folds[-k])
    train_ds <- ds(train_indices)
    train_dl <- train_ds %>% torch::dataloader(batch_size = control$batch.size, shuffle = TRUE)
    valid_ds <- ds(valid_indices)
    valid_dl <- valid_ds %>% torch::dataloader(batch_size = control$batch.size, shuffle = FALSE)
    if (method == "MAP") {
      model <- nn_SPQR_MAP(V, # MAP estimation using one of the three priors
                           control$dropout,
                           control$batchnorm,
                           activation,
                           prior,
                           hyperpar$a_sigma,
                           hyperpar$b_sigma,
                           hyperpar$a_lambda,
                           hyperpar$b_lambda,
                           device=device)
    } else {
      model <- nn_SPQR_MLE(V, # MLE estimation
                           control$dropout,
                           control$batchnorm,
                           activation)
    }
    model$to(device=device)
    Btrain <- torch_tensor(Btotal[train_indices,], device=device)
    Bvalid <- torch_tensor(Btotal[valid_indices,], device=device)

    optimizer <- torch::optim_adam(model$parameters, lr = control$lr)
    counter <- 0

    if (verbose) cat(sprintf("Starting fold: %d/%d\n", k, K))

    last_valid_loss <- Inf
    last_train_loss <- Inf
    for (epoch in 1:control$epochs) {

      model$train()
      train_losses <- c()

      coro::loop(for (b in train_dl) {
        optimizer$zero_grad()
        result <- model(b$x)
        indices <- b$index
        nloglik <- nll.loss(indices=indices, basis=Btrain, coefs=result$output)
        loss <- nloglik - result$logprior

        loss$backward()
        optimizer$step()

        train_losses <- c(train_losses, loss$item())

      })

      model$eval()
      valid_losses <- c()

      coro::loop(for (b in valid_dl) {
        result <- model(b$x)
        indices <- b$index
        nloglik <- nll.loss(indices=indices, basis=Bvalid, coefs=result$output)
        loss <- nloglik - result$logprior

        valid_losses <- c(valid_losses, loss$item())

      })
      train_loss <- mean(train_losses)
      valid_loss <- mean(valid_losses)
      if (verbose) {
        if (epoch == 1 || epoch %% control$print.every.epochs == 0) {
          cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch,
                      train_loss, valid_loss))
        }
      }
      if (!is.null(control$early.stopping.epochs)) {
        if (valid_loss < last_valid_loss) {
          last_valid_loss <- valid_loss
          last_train_loss <- train_loss
          counter <- 0
        } else {
          counter <- counter + 1
          if (counter >= control$early.stopping.epochs) {
            if (verbose) {
              cat(sprintf("Stopping... Best epoch: %d\n", epoch))
              cat(sprintf("Final loss: training: %3f, validation: %3f\n\n",
                          last_train_loss, last_valid_loss))
            }
            break
          }
        }
      } else {
        last_valid_loss <- valid_loss
        last_train_loss <- train_loss
      }
    }
    cv_losses[k] <- last_valid_loss
  }
  out <- list(control=control, cve=mean(cv_losses), folds=folds)
  invisible(out)
}
