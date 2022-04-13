#' @title Fitting SPQR by MCMC method
#'
#' @importFrom stats rgamma
#' @importFrom progressr handlers progressor
SPQR.MCMC <-
  function(X, Y, n.knots, n.hidden, activation, prior, hyperpar, control,
           verbose=TRUE, seed=seed)
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
                control=control,
                X=t(X),
                Y=Y)
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
