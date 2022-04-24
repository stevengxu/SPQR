#' @title Fitting SPQR models
#' @description
#' Main function of the package. Fits SPQR using the maximum likelihood estimation (MLE), maximum \emph{a posterior} (MAP) or
#' Markov chain Monte Carlo (MCMC) method. Returns an object of S3 class \code{SPQR}.
#'
#' @docType package
#' @useDynLib SPQR
#'
#' @name SPQR
#' @param X The covariate matrix (without intercept column)
#' @param Y The response vector.
#' @param n.knots The number of basis functions. Default: 10.
#' @param n.hidden A vector specifying the number of hidden neurons in each hidden layer. Default: 10.
#' @param activation The hidden layer activation. Either \code{"tanh"} (default) or \code{"relu"}.
#' @param method Method for estimating SPQR. One of \code{"MLE"}, \code{"MAP"} (default) or \code{"MCMC"}.
#' @param prior The prior model for variance hyperparameters. One of \code{"GP"}, \code{"ARD"} (default) or \code{"GSM"}.
#' @param hyperpar A list of named hyper-prior hyperparameters to use instead of the default values, including
#'   \code{a_lambda}, \code{b_lambda}, \code{a_sigma} and \code{b_sigma}. The default value is 0.001 for all four
#'   hyperparameters.
#' @param control A list of named and method-dependent parameters that allows finer
#'  control of the behavior of the computational approaches.
#'
#' 1. Parameters for MLE and MAP methods
#'
#' \itemize{
#'   \item \code{use.GPU} If \code{TRUE} GPU computing will be used if \code{torch::cuda_is_available()} returns \code{TRUE}. Default: \code{FALSE}.
#'   \item \code{lr} The learning rate used by the Adam optimizer, i.e., \code{torch::optim_adam()}.
#'   \item \code{dropout} A length two vector specifying the dropout probabilities in the input and hidden layers respectively. The default is \code{c(0,0)} indicating no dropout.
#'   \item \code{batchnorm} If \code{TRUE} batch normalization will be used after each hidden activation.
#'   \item \code{epochs} The number of passes of the entire training dataset in gradient descent optimization. If \code{early.stopping.epochs} is used then this is the maximum number of passes. Default: 200.
#'   \item \code{batch.size} The size of mini batches for gradient calculation. Default: 128.
#'   \item \code{valid.pct} The fraction of data used as validation set. Default: 0.2.
#'   \item \code{early.stopping.epochs} The number of epochs before stopping if the validation loss does not decrease. Default: 10.
#'   \item \code{print.every.epochs} The number of epochs before next training progress in printed. Default: 10.
#'   \item \code{save.path} The path to save the fitted torch model. By default a folder named \code{"SPQR_model"} is created in the current working directory to store the model.
#'   \item \code{save.name} The name of the file to save the fitted torch model. Default is \code{"SPQR.model.pt"}.
#' }
#'
#' 2. Parameters for MCMC method
#'
#' These parameters are similar to those in \code{rstan::stan()}. Detailed explanations can be found in
#' the Stan reference manual.
#'
#' \itemize{
#'   \item \code{algorithm} The sampling algorithm; \code{"HMC"}: Hamiltonian Monte Carlo with dual-averaging, \code{"NUTS"}: No-U-Turn sampler (default).
#'   \item \code{iter} The number of MCMC iterations (including warmup). Default: 2000.
#'   \item \code{warmup} The number of warm-up/burn-in iterations for step-size and mass matrix adaptation. Default: 500.
#'   \item \code{thin} The number of iterations before saving next post-warmup samples. Default: 1.
#'   \item \code{stepsize} The discretization interval/step-size \eqn{\epsilon} of leap-frog integrator. Default is \code{NULL} which indicates that it will be adaptively selected during warm-up iterations.
#'   \item \code{metric} The type of mass matrix; \code{"unit"}: diagonal matrix of ones, \code{"diag"}: diagonal matrix with positive diagonal entries estimated during warmup iterations (default), \code{"dense"}: a dense, symmetric positive definite matrix with entries estimated during warm-up iterations.
#'   \item \code{delta} The target Metropolis acceptance rate. Default: 0.9.
#'   \item \code{max.treedepth} The maximum tree depth in NUTS. Default: 6.
#'   \item \code{int.time} The integration time in HMC. The number of leap-frog steps is calculated as \eqn{L_{\epsilon}=\lfloor t/\epsilon\rfloor}. Default: 0.3.
#' }
#'
#' @param normalize If \code{TRUE}, all covariates will be normalized to take values between [0,1].
#' @param verbose If \code{TRUE} (default), training progress will be printed.
#' @param seed Random number generation seed.
#' @param ... other parameters to pass to \code{control}.
#'
#' @return An object of class \code{"SPQR"}.
#'
#'
#' @importFrom torch `%>%` torch_tensor
#' @importFrom stats rgamma
#' @importFrom progressr handlers progressor
#'
#' @references Xu SG, Reich BJ (2021). \emph{Bayesian Nonparametric Quantile Process Regression and Estimation of Marginal Quantile Effects.} Biometrics. \href{doi.org/10.1111/biom.13576}{doi:10.1111/biom.13576}
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
#' ## plot estimated PDF
#' plotEstimator(fit, type = "PDF", X = 0)
#'
#' @export
SPQR <-
  function(X, Y, n.knots = 10, n.hidden = 10, activation = c("tanh","relu","sigmoid"),
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
  if (sum(is.na(X)) > 0) stop("`X` cannot have missing values")
  if (!is.numeric(try(sum(X[1,]),silent=TRUE))) stop("`X` cannot have non-numeric values")

  if (!is.matrix(X)) X <- as.matrix(X) # data.frame case

  ny <- NCOL(Y)
  if (is.matrix(Y) && ny == 1) Y <- drop(Y) # treat 1D matrix as vector
  if (NROW(Y) != n) stop("incompatible dimensions")
  # normalize all covariates
  if (normalize) {
    X.range <- apply(X,2,range)
    .X <- apply(X,2,function(x){
      (x - min(x)) / (max(x) - min(x))
    })
    Y.range <- range(Y)
    .Y <- (Y - Y.range[1])/diff(Y.range)
  } else {
    .X <- X; .Y <- Y
    if (min(.Y) < 0 || max(.Y) > 1) stop("`Y` must be between 0 and 1")
  }
  control <- .check.control(control, method, ...)
  hyperpar <- .update.hyperpar(hyperpar)
  if (method == "MCMC") {
    out <- SPQR.MCMC(X=.X, Y=.Y, n.knots=n.knots, n.hidden=n.hidden,
                     activation=activation, prior=prior, hyperpar=hyperpar,
                     control=control, verbose=verbose, seed=seed)
    out$method <- method
  } else {
    out <- SPQR.ADAM(X=.X, Y=.Y, n.knots=n.knots, n.hidden=n.hidden,
                     activation=activation, method=method, prior=prior,
                     hyperpar=hyperpar, control=control, verbose=verbose,
                     seed=seed)
  }
  out$X <- X; out$Y <- Y
  if (normalize) {
    out$normalize$X <- X.range
    out$normalize$Y <- Y.range
  }
  class(out) <- "SPQR"
  invisible(out)
}

## Internal function for fitting SPQR using MLE and MAP methods

SPQR.ADAM <- function(X, Y, n.knots, n.hidden, activation, method, prior,
                      hyperpar, control, verbose, seed)
{

  self <- NULL
  use.GPU <- control$use.GPU
  cuda <- torch::cuda_is_available()
  if(use.GPU && !cuda){
    warning('GPU acceleration not available, using CPU')
    use.GPU <- FALSE
  } else if (!use.GPU && cuda){
    message('GPU acceleration is available through `use.GPU=TRUE`')
  }
  device <- if (use.GPU) "cuda" else "cpu"
  control$use.GPU <- use.GPU

  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }
  V <- c(ncol(X), n.hidden, n.knots)
  # Define dataset and dataloader
  ds <- torch::dataset(
    initialize = function(indices) {
      self$x <- torch_tensor(X[indices,,drop=FALSE], device=device)
      self$y <- torch_tensor(Y[indices], device=device)
    },

    .getbatch = function(i) {
      list(x = self$x[i,], y = self$y[i], index = i)
    },

    .length = function() {
      self$y$size()[[1]]
    }
  )
  N <- nrow(X)
  if (control$valid.pct > 0) {
    valid_indices <- sample(1:N, size = floor(N*control$valid.pct))
    train_indices <- setdiff(1:N, valid_indices)
    train_ds <- ds(train_indices)
    train_dl <- train_ds %>% torch::dataloader(batch_size=control$batch.size, shuffle=TRUE)
    valid_ds <- ds(valid_indices)
    valid_dl <- valid_ds %>% torch::dataloader(batch_size=control$batch.size, shuffle=FALSE)
  } else {
    control$early.stopping.epochs <- Inf
    train_indices <- 1:N
    train_ds <- ds(train_indices)
    train_dl <- train_ds %>% torch::dataloader(batch_size=control$batch.size, shuffle=FALSE)
  }

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

  # Computing the basis and converting it to a tensor beforehand to save
  # computational time; this is used every iteration for computing loss
  Btotal <- .basis(Y, n.knots)
  Btrain <- torch_tensor(Btotal[train_indices,], device=device)
  if (control$valid.pct > 0) Bvalid <- torch_tensor(Btotal[valid_indices,], device=device)

  # Define custom loss function
  nll.loss = function(indices, basis, coefs) {
    loglik <- basis[indices,]$mul(coefs)$sum(2)$log()$sum()
    return(-loglik)
  }

  optimizer <- torch::optim_adam(model$parameters, lr = control$lr)
  counter <- 0
  if(!dir.exists(control$save.path)) dir.create(control$save.path)

  save_name <- file.path(control$save.path, control$save.name)
  last_valid_loss <- Inf
  last_train_loss <- Inf
  time.start <- Sys.time()
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

    if (control$valid.pct > 0) {
      model$eval()
      valid_losses <- c()

      coro::loop(for (b in valid_dl) {

        result <- model(b$x)
        indices <- b$index
        nloglik <- nll.loss(indices=indices, basis=Bvalid, coefs=result$output)
        loss <- nloglik - result$logprior

        valid_losses <- c(valid_losses, loss$item())

      })
    }
    train_loss <- mean(train_losses)
    if (control$valid.pct > 0) valid_loss <- mean(valid_losses)
    if (verbose) {
      if (epoch == 1 || epoch %% control$print.every.epochs == 0) {
        cat(sprintf("Loss at epoch %d: training: %3f", epoch,
                    train_loss))
        if (control$valid.pct > 0) cat(sprintf(", validation: %3f\n", valid_loss))
        else cat("\n")
      }
    }
    if (is.finite(control$early.stopping.epochs)) {
      if (valid_loss < last_valid_loss) {
        torch::torch_save(model, save_name)
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
      if (control$valid.pct > 0) last_valid_loss <- valid_loss
      last_train_loss <- train_loss
    }
  }
  time.total <- difftime(Sys.time(), time.start, units='mins')
  # load best model
  best.model <- torch::torch_load(save_name)$cpu()

  config <- list(n.knots=n.knots,
                 n.hidden=n.hidden,
                 activation=activation)
  if (method == "MAP") {
    config$prior <- prior
    config$hyperpar <- hyperpar
  }
  if (control$valid.pct > 0) loss <- list(train = last_train_loss, validation = last_valid_loss)
  else loss <- list(train = last_train_loss)
  out <- list(model=best.model,
              loss=loss,
              time=time.total,
              method=method,
              config=config,
              control=control)
  return(out)
}
## End of ADAM method

## Internal function for fitting SPQR using MCMC method

SPQR.MCMC <-
  function(X, Y, n.knots, n.hidden, activation, prior, hyperpar, control,
           verbose, seed)
{
  X <- t(X)
  B <- t(.basis(Y, n.knots)) # M-spline basis
  nvar <- nrow(X)
  V <- c(nvar, n.hidden, n.knots) # Number of nodes for each layer
  n.layers <- length(V) - 1 # number of layers
  .params <- list(V=V, activation=activation)
  npar <- 0
  for (l in 1:n.layers) npar <- npar + (V[l] + 1)*V[l+1]

  asig <- hyperpar$a_sigma
  bsig <- hyperpar$b_sigma
  alam <- hyperpar$a_lambda
  blam <- hyperpar$b_lambda

  metric <- control$metric
  eps <- control$stepsize

  if (metric == "dense") {
    init.stepsize <- rcpp_init_stepsize_dense
    sampling <- if (control$algorithm == "NUTS") rcpp_nuts_dense else rcpp_hmc_dense
  } else {
    init.stepsize <- rcpp_init_stepsize_diag
    sampling <- if (control$algorithm == "NUTS") rcpp_nuts_diag else rcpp_hmc_diag
  }

  if (control$algorithm == "NUTS") {
    const.var <- "treedepth"
    const <- control$max.treedepth
  } else {
    const.var <- "num.steps"
    const <- control$int.time
  }

  # Adapt stepsize?
  adapt.eps <- is.null(eps)
  if (adapt.eps) {
    gamma <- control$gamma
    delta <- control$delta
    kappa <- control$kappa
    t0 <- control$t0
  }
  # Adapt mass matrix?
  adapt.M <- metric != "unit" && adapt.eps
  # No need to warmup if neither mass matrix nor stepsize is adapted
  if (!adapt.M && !adapt.eps) control$warmup <- 0
  # The inverse metric `Minv` aims to approximate the covariance matrix
  # of the parameters. It is always initialized to unit diagonal.
  Misqrt <- Minv <- rep(1, npar)
  if (adapt.M) {
    if (metric == "dense") Misqrt <- Minv <- diag(npar)
    window.adapter <- initialize_window_adapter(control$warmup, control)
    # Initialize variance adaptation placeholders
    wns <- 0
    wm <- rep(0, npar)  # First moment
    # Second moment
    if(metric == "diag")
      wm2 <- rep(0, npar)
    else
      wm2 <- matrix(0, npar, npar)
  }

  # Initial values
  theta <- .init.W(V)

  .params$sigma <- rep(1, n.layers) # Global layerwise scale
  .params$lambda <- vector("list", n.layers) # Local unitwise scale for W
  for (l in 1:n.layers) .params$lambda[[l]] <- rep(1, V[l]+1)

  nsave <- floor((control$iter-control$warmup)/control$thin)
  # Results placeholders
  model <- vector("list", nsave)
  ll.mat <- matrix(0, nrow=nsave, ncol=length(Y))
  chain.info <- lapply(1:3,function(i){
    numeric(nsave)})
  names(chain.info) <- c("accept.ratio", const.var, "divergent")


  if (adapt.eps) {
    Hbar <- 0
    eps <- init.stepsize(theta, Minv, Misqrt, X, B, .params)
    log.epsbar <- log(eps)
    mu <- log(10*eps)
  }

  # Start of MCMC chain
  time.start <- Sys.time()
  if (verbose) {
    message('')
    message(paste('Starting', control$algorithm, 'at', time.start))
  }

  for (i in 1:control$iter) {
    if (i == 1 && verbose) {
      handlers("progress")
      handlers(global = TRUE)
      pb <- progressor(control$warmup)
    }
    info <- list(0, 0)
    names(info) <- c(eval(const.var), "divergent")

    draw <- sampling(theta, X, B, .params, eps, Minv, Misqrt, const, info)
    theta <- draw$theta
    accept.prob <- draw$accept.prob
    W <- .theta2W(theta, V)

    # Gibbs sampler for scales
    for (l in 1:n.layers) {
      H <- ifelse(l==1, 1, V[l])
      if (prior == "GSM") {
        # All inputs (including bias) have a layerwise global scale
        # In addition, each input is associated with a input-specific local scale

        # Update global scale
        aa <- asig + V[l+1]*(V[l]+1)/2
        bb <- bsig + (sum((t(W$W[[l]])/.params$lambda[[l]][-1])^2)+sum((W$b[[l]]/.params$lambda[[l]][1])^2))/2
        .params$sigma[l] <- 1/sqrt(rgamma(1,aa,bb))

        # Update local scale for weights
        cc <- alam + V[l+1]/2
        # Separate local scale for each input
        for (j in 1:V[l]) {
          dd <- blam+sum((W$W[[l]][,j]/.params$sigma[l])^2)/2
          .params$lambda[[l]][j+1] <- 1/sqrt(rgamma(1,cc,dd))
        }
        # Update local scale for biases
        ff <- blam + sum((W$b[[l]]/.params$sigma[l])^2)/2
        .params$lambda[[l]][0] <- 1/sqrt(rgamma(1,cc,ff))
      } else if (prior == "ARD" && l == 1) {
        # Update local scale for weights
        cc <- alam + V[2]/2
        # Separate local scale for each input
        for (j in 1:V[1]) {
          dd <- blam+sum((W$W[[1]][,j])^2)/2
          .params$lambda[[1]][j+1] <- 1/sqrt(rgamma(1,cc,dd))
        }
        # Update local scale for biases
        ff <- blam + sum((W$b[[1]])^2)/2
        .params$lambda[[1]][1] <- 1/sqrt(rgamma(1,cc,ff))
      } else {
        # Weigths and biases have block-wise variances

        # Update scale for weights
        cc <- alam + V[l+1]*V[l]/2
        dd <- blam/H + sum(W$W[[l]]^2)/2
        .params$lambda[[l]][-1] <- 1/sqrt(rgamma(1,cc,dd))

        # Update scale for biases
        ee <- alam + V[l+1]/2
        ff <- blam + sum((W$b[[l]])^2)/2
        .params$lambda[[l]][1] <- 1/sqrt(rgamma(1,ee,ff))
      }
    }# end of gibbs update

    if (i <= control$warmup) {
      if (verbose) pb(message = "Warming up...")
      if (adapt.eps) {
        # Dual avergaing to adapt step size
        eta <- 1 / (i + t0)
        Hbar <- (1 - eta) * Hbar + eta * (delta - accept.prob)
        log.eps <- mu - sqrt(i) * Hbar / gamma
        eta <- i^-kappa
        log.epsbar <- (1 - eta) * log.epsbar + eta * log.eps
        eps <- exp(log.eps)
      } # End of step size adaptation

      if (adapt.M) {
        # Adaptation of mass matrix
        if (adaptation_window(window.adapter)) {
          wns <- wns + 1
          wdelta <- theta - wm
          wm <- wm + wdelta / wns
          if (metric == "diag") wm2 <- wm2 + (theta - wm) * wdelta
          else wm2 <- wm2 + tcrossprod(theta - wm, wdelta)
        }

        if (end_adaptation_window(window.adapter)) {
          window.adapter <- compute_next_window(window.adapter)
          Minv <- wm2 / (wns - 1)
          if (metric == "diag")
            Minv <- (wns / (wns + 5)) * Minv + 1e-3 * (5 / (wns + 5))
          else
            Minv <- (wns / (wns + 5)) * Minv + 1e-3 * (5 / (wns + 5)) * diag(npar)
          if (!is.finite(sum(Minv))) {
            warning("Non-finite estimates in mass matrix adaptation ",
                    "-- reverting to unit metric.")
            Minv <- rep(1, len = npar)
          }
          # Update symmetric square root
          Misqrt <- if(metric == "diag") sqrt(Minv) else chol(Minv)
          # Find new reasonable eps since it can change dramatically when mass
          # matrix updates
          eps <- init.stepsize(theta, Minv, Misqrt, X, B, .params)
          mu  <- log(10 * eps)
          # Reset the running variance calculation
          wns <- 0
          wm <- rep(0, len=npar)
          if (metric == "diag") wm2 <- rep(0, npar)
          else wm2 <- matrix(0, npar, npar)
        }
        window.adapter$window.counter <- window.adapter$window.counter + 1
      } # End of mass matrix adaptation
    } else {
      # Fix stepsize after warmup
      if (i == control$warmup + 1) {
        if (adapt.eps) eps <- exp(log.epsbar)
        if (verbose) {
          handlers(global = TRUE)
          pb <- progressor(control$iter - control$warmup)
        }
        isave <- 1 # Initialize saving index
      }
      if (verbose) pb(message = "Sampling...")
      if ((i - control$warmup) %% control$thin == 0) {
        model[[isave]] <- W
        ll.mat[isave,] <- loglik_vec(theta, X, B, .params)

        chain.info$accept.ratio[isave] <- accept.prob
        chain.info[[2]][isave] <- info[[1]]
        chain.info$divergent[isave] <- info[[2]]
        isave <- isave + 1
      }
    }
  } # End of MCMC loop
  if (verbose) handlers(global = FALSE)
  chain.info$stepsize <- eps
  if (!adapt.eps) delta <- NA
  chain.info$delta <- delta
  chain.info$loglik <- ll.mat
  time.total <- difftime(Sys.time(), time.start, units='mins')
  if (verbose) message(paste0("Elapsed Time: ", sprintf("%.1f", time.total), ' minutes'))

  config <- list(n.knots=n.knots,
                 n.hidden=n.hidden,
                 activation=activation,
                 prior=prior,
                 hyperpar=hyperpar)

  out <- list(model=model,
              time=time.total,
              chain.info=chain.info,
              config=config,
              control=control)
  return(out)
}

## Initialize network using xavier's uniform rule
.init.W <- function(V) {

  n.layers <- length(V) - 1
  theta <- {}
  for (l in 1:n.layers) {
    W <- sqrt(6)/sqrt(V[l]+1+V[l+1])*stats::runif(V[l]*V[l+1],-1,1)
    b <- sqrt(6)/sqrt(V[l]+1+V[l+1])*stats::runif(V[l+1],-1,1)
    theta <- c(theta, c(W,b))
  }
  return(theta)
}

## Extract posterior samples of weights and bias
.theta2W <- function(theta, V) {

  n.layers <- length(V) - 1

  .W <- vector("list", n.layers)
  .b <- vector("list", n.layers)

  end <- 0
  for (l in 1:n.layers) {
    start <- end + 1
    end <- start + V[l]*V[l+1] - 1
    .W[[l]] <- theta[start:end]
    dim(.W[[l]]) <- c(V[l+1],V[l])
    start <- end + 1
    end <- start + V[l+1] - 1
    .b[[l]] <- theta[start:end]
  }

  return(list(W=.W, b=.b))
}

initialize_window_adapter <- function(warmup, control) {
  init.buffer <- control$init.buffer
  term.buffer <- control$term.buffer
  base.window <- control$base.window

  if (warmup < (init.buffer + term.buffer + base.window)) {
    warning("There aren't enough warmup iterations to fit the ",
            "three stages of adaptation as currently configured.")
    init.buffer <- 0.15 * warmup
    term.buffer <- 0.1 * warmup
    base.window <- warmup - (init.buffer + term.buffer)
    message("Reducing each adaptation stage to 15%/75%/10% of ",
            "the given number of warmup iterations:\n",
            paste("init.buffer =", init.buffer),
            paste("base.window =", base.window),
            paste("term.buffer =", term.buffer))
  }
  adapter <- list(
    warmup = warmup,
    window.counter = 0,
    init.buffer = init.buffer,
    term.buffer = term.buffer,
    base.window = base.window,
    window.size = base.window,
    next.window = init.buffer + term.buffer
  )
  invisible(adapter)
}

## Check if iteration is within adaptation window

adaptation_window <- function(adapter) {
  c1 <- adapter$window.counter >= adapter$init.buffer
  c2 <- adapter$window.counter <= adapter$warmup - adapter$term.buffer
  c3 <- adapter$window.counter < adapter$warmup
  return(c1 && c2 && c3)
}

## Check if iteration is at the end of adaptation window

end_adaptation_window <- function(adapter) {
  c1 <- adapter$window.counter == adapter$next.window
  c2 <- adapter$window.counter < adapter$warmup
  return(c1 && c2)
}

## Compute the next window size in mass matrix adaptation

compute_next_window <- function(adapter) {

  next.window <- adapter$next.window
  warmup <- adapter$warmup
  term.buffer <- adapter$term.buffer
  window.size <- adapter$window.size

  if(next.window == (warmup - term.buffer))
    return(adapter)

  window.size <- window.size * 2
  adapter$window.size <- window.size
  next.window <- adapter$window.counter + window.size

  if(next.window == (warmup - term.buffer)) {
    adapter$next.window <- next.window
    return(adapter)
  }

  next.window_boundary <- next.window + 2 * window.size
  if(next.window_boundary >= warmup - term.buffer)
    next.window <- warmup - term.buffer

  adapter$next.window <- next.window
  invisible(adapter)
}

## End of MCMC method
