sourceCpp("../src/advanced_nuts.cpp")
sourceCpp("../src/static_hmc.cpp")
sourceCpp("../src/find_epsilon.cpp")

SPQR.mcmc <- 
  function(params, X, Y, verbose = TRUE)
{
  X <- t(X)
  B <- t(.basis(Y, params$n.knots)) # M-spline basis
  nvar <- nrow(X)
  V <- c(nvar, params$n.hidden, params$n.knots) # Number of nodes for each layer
  n.layers <- length(V) - 1 # number of layers
  bnn.params <- list(V=V, activation=params$activation)
  npar <- 0
  for (l in 1:n.layers) npar <- npar + (V[l] + 1)*V[l+1]
  
  control <- .update.control(params$control)
  metric <- match.arg(control$metric, c("unit","diag","dense"))
  
  eps <- control$stepsize
  
  if (params$sampler == "NUTS") {
    const.var <- "treedepth"
    const <- control$max.treedepth
    sampler <- "advanced_nuts"
  } else {
    const.var <- "num.steps"
    const <- control$int.time
    sampler <- "static_hmc"
  }
  sampling <- eval(parse(text = paste(sampler, metric, sep = "_")))
  init.stepsize <- eval(parse(text = paste("init_stepsize", metric, sep = "_")))
  
  # Dual Averaging arguments
  adapt.eps <- is.null(eps)
  if (adapt.eps) {
    gamma <- control$gamma
    delta <- control$delta
    kappa <- control$kappa
    t0 <- control$t0
  }
  # Mass matrix adaptation arguments
  adapt.M <- metric != "unit" && adapt.eps 
  # The inverse metric `Minv` aims to approximate the covariance matrix
  # of the parameters. It is always initialized to unit diagonal.
  Misqrt <- Minv <- rep(1, npar)
  if (adapt.M) {
    if (metric == "dense") Misqrt <- Minv <- diag(npar)
    window.adapter <- initialize_window_adapter(params$warmup, control)
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
  
  bnn.params$sigma_W <- rep(1, n.layers)
  bnn.params$sigma_b <- rep(1, n.layers)
  bnn.params$lambda_W <- vector("list", n.layers)
  for (l in 1:n.layers) bnn.params$lambda_W[[l]] <- rep(1, V[l])
  bnn.params$lambda_b <- rep(1, n.layers)
   
  nsave <- floor((params$iter-params$warmup)/params$thin)
  # Results placeholders 
  model <- vector("list", nsave)
  ll.mat <- matrix(0, nrow=nsave, ncol=length(Y))
  chain.info <- lapply(1:3,function(i){
    numeric(nsave)})
  names(chain.info) <- c("accept.ratio", const.var, "divergent")
  
  
  if (adapt.eps) {
    Hbar <- 0
    #eps <- find_reasonable_epsilon(theta[1,], Minv, Misqrt, X, B, param)
    eps <- init.stepsize(theta, Minv, Misqrt, X, B, bnn.params)
    log.epsbar <- log(eps)
    mu <- log(10*eps)
  }
  
  # Start of MCMC chain
  if (verbose) {
    message('')
    message(paste('Starting', params$sampler, 'at', time.start <- Sys.time()))
  }

  for (i in 1:params$iter) {
    if (i == 1 && verbose) {
      handlers("progress")
      handlers(global = TRUE)
      pb <- progressor(params$warmup)
    }
    
    info <- list(0, 0)
    names(info) <- c(eval(const.var), "divergent")
    
    draw <- sampling(theta, X, B, bnn.params, eps, Minv, Misqrt, const, info)
    theta <- draw$theta
    accept.prob <- draw$accept.prob
    W <- .theta2W(theta, V)
    
    # Gibbs sampler for scales
    bnn.params <- .gibbs(bnn.params, W, params$prior, 
                         params$hyperpar$a_sigma, params$hyperpar$b_sigma,
                         params$hyperpar$a_lambda, params$hyperpar$b_lambda)
                
    if (i <= params$warmup) {
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
          eps <- init.stepsize(theta, Minv, Misqrt, X, B, bnn.params)
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
      if (i == params$warmup + 1) {
        if (adapt.eps) eps <- exp(log.epsbar)
        if (verbose) {
          handlers(global = TRUE)
          pb <- progressor(params$iter - params$warmup)
        }
        isave <- 1 # Initialize saving index
      }
      if (verbose) pb(message = "Sampling...")
      if ((i - params$warmup) %% params$thin == 0) {
        model[[isave]] <- W
        ll.mat[isave,] <- loglik_vec(theta, X, B, bnn.params)
        
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
  params$control <- control
  out <- list(model=model,
              time=time.total,
              chain.info=chain.info, 
              params=params,
              X=t(X), 
              Y=Y)
  return(out)
}

.gibbs <- function(params, W, prior, a_sigma, b_sigma, a_lambda, b_lambda) {
  
  V <- params[["V"]]
  num_layers <- length(V) - 1
  .W <- W$W
  .b <- W$b
  if (prior == "ARD" || prior == "GSM") {
    # local scale shouldn't be touched for "GP" prior
    lambda_W <- params$lambda_W
    lambda_b <- params$lambda_b
  }
  sigma_W <- params$sigma_W
  sigma_b <- params$sigma_b
  
  for (l in 1:num_layers) {
    H <- ifelse(l==1, 1, V[l])
    if (prior == "GSM") {
      # All inputs (including bias) have a layerwise global scale
      # In addition, each input is associated with a input-specific local scale
      
      # Update global scale
      aa <- a_sigma + V[l+1]*(V[l]+1)/2
      bb <- b_sigma + (sum((t(.W[[l]])/lambda_W[[l]])^2)+sum((.b[[l]]/lambda_b[l])^2))/2
      sigma_b[l] <- sigma_W[l] <- 1/sqrt(rgamma(1,aa,bb))
      
      # Update local scale for weights
      cc <- a_lambda + V[l+1]/2
      # Separate local scale for each input
      for (j in 1:V[l]) {
        dd <- b_lambda+sum((.W[[l]][,j]/sigma_W[l])^2)/2
        lambda_W[[l]][j] <- 1/sqrt(rgamma(1,cc,dd))
      }
      # Update local scale for biases
      ff <- b_lambda + sum((.b[[l]]/sigma_b[l])^2)/2
      lambda_b[l] <- 1/sqrt(rgamma(1,cc,ff))
    } else if (prior == "ARD" && l == 1) {
      # Update local scale for weights
      cc <- a_sigma + V[2]/2
      # Separate local scale for each input
      for (j in 1:V[1]) {
        dd <- b_sigma+sum((.W[[1]][,j])^2)/2
        lambda_W[[1]][j] <- 1/sqrt(rgamma(1,cc,dd))
      }
      # Update local scale for biases
      ff <- b_sigma + sum((.b[[1]])^2)/2
      sigma_b[1] <- 1/sqrt(rgamma(1,cc,ff))
    } else {
      # Weigths and biases have block-wise variances
      
      # Update scale for weights
      cc <- a_sigma + V[l+1]*V[l]/2
      dd <- b_sigma/H + sum(.W[[l]]^2)/2
      sigma_W[l] <- 1/sqrt(rgamma(1,cc,dd))
      
      # Update scale for biases
      ee <- a_sigma + V[l+1]/2
      ff <- b_sigma + sum((.b[[l]])^2)/2
      sigma_b[l] <- 1/sqrt(rgamma(1,ee,ff))
    }
  }
  if (prior == "ARD" || prior == "GSM") {
    params$lambda_W <- lambda_W
    params$lambda_b <- lambda_b
  }
  params$sigma_W <- sigma_W
  params$sigma_b <- sigma_b
  return(params)
}

.coefs <- function(W, X, activation = "tanh") {
  .W <- W$W
  .b <- W$b
  n.layers <- length(W)
  A <- X
  for (l in 1:n.layers) {
    A <- .W[[l]]%*%A + .b[[l]]
    if (l < n.layers) {
      if (activation == "tanh")
        A <- base::tanh(A)
      else
        A <- pmax(A,0)
    }
  }
  enn <- exp(A)
  p <- sweep(enn, 2, colSums(enn), '/')
  return(t(p))
}

## Update the control list
.update.control <- function(control) {
  default <- list(
    gamma = 0.05,
    delta = 0.9,
    kappa = 0.75,
    t0 = 10,
    init.buffer = 75,
    term.buffer = 50,
    base.window = 25,
    stepsize = NULL,
    metric = "diag",
    max.treedepth = 6,
    int.time = 0.3
  )
  if (!is.null(control)) {
    for (i in names(control))
      default[[i]] <- control[[i]]
  }
  invisible(default)
}

## Initialize network using xavier's uniform rule
.init.W <- function(V) {
  
  n.layers <- length(V) - 1
  theta <- {}
  for (l in 1:n.layers) {
    W <- sqrt(6)/sqrt(V[l]+1+V[l+1])*runif(V[l]*V[l+1],-1,1)
    b <- sqrt(6)/sqrt(V[l]+1+V[l+1])*runif(V[l+1],-1,1)
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