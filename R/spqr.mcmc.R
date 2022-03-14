sourceCpp("../src/advanced_nuts.cpp")
sourceCpp("../src/static_hmc.cpp")
sourceCpp("../src/find_epsilon.cpp")

spqr.mcmc.train <- 
  function(params, X, y, seed = NULL, verbose = 2)
{
  if (!is.null(seed)) set.seed(seed)

  now <- function() {
    paste0('[', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ']')
  }  
    
  X <- t(X)
  B <- sp.basis(y, params[["n.knots"]]) # M-spline basis
  iter <- params[["iter"]]; warmup <- params[["warmup"]]; thin <- params[["thin"]]
  nvar <- nrow(X)
  V <- c(nvar, params[["n.hidden"]], params[["n.knots"]]) # Number of nodes for each layer
  num_layers <- length(V) - 1
  bnn.params <- list(V=V, activation = params[["activation"]])
  npar <- 0
  for (l in 1:num_layers) {
    npar <- npar + (V[l] + 1)*V[l+1]
  }
  
  control <- update.hmc.control(params[["hmc.control"]])
  metric <- control$metric
  if (metric %notin% c("unit","diag","dense"))
    abort("'metric' must be one of c('unit','diag','dense')")
  eps <- control$stepsize
  
  if (params[["sampler"]] == "NUTS") {
    constraint_name <- "treedepth__"
    constraint <- control$max_tree_depth
    sampler <- "advanced_nuts"
  } else {
    constraint_name <- "num_steps__"
    constraint <- control$int_time
    sampler <- "static_hmc"
  }
  sampling <- eval(parse(text = paste(sampler, metric, sep = "_")))
  init_stepsize <- eval(parse(text = paste("init_stepsize", metric, sep = "_")))
  
  # Dual Averaging arguments
  adapt_Eps <- is.null(eps)
  if (adapt_Eps) {
    gamma <- control$adapt_gamma
    delta <- control$adapt_delta
    kappa <- control$adapt_kappa
    t0 <- control$adapt_t0
  }
  # Mass matrix adaptation arguments
  adapt_Mass <- metric != "unit" && adapt_Eps 
  # The inverse metric `Minv` aims to approximate the covariance matrix
  # of the parameters. It is always initialized to unit diagonal.
  Misqrt <- Minv <- rep(1, npar)
  if (adapt_Mass) {
    if (metric == "dense")
      Misqrt <- Minv <- diag(npar)
    
    window_adapter <- initialize_window_adapter(warmup, control)
    # Initialize variance adaptation placeholders
    welford_num_samples <- 0
    welford_m <- rep(0, npar)  # First moment
    # Second moment
    if(metric == "diag")
      welford_m2 <- rep(0, npar)
    else
      welford_m2 <- matrix(0, npar, npar)
  }
  
  # Initial values
  theta <- nn.init.xavier.uniform(V)
  
  bnn.params$sigma_W <- rep(1, num_layers)
  bnn.params$sigma_b <- rep(1, num_layers)
  bnn.params$lambda_W <- vector("list", num_layers)
  for (l in 1:num_layers) 
    bnn.params$lambda_W[[l]] <- rep(1, V[l])
  
  bnn.params$lambda_b <- rep(1, num_layers)
   
  
  nsave <- floor((iter-warmup)/thin)
  # Results placeholders 
  WB_ <- vector("list", nsave)
  loglik_mat_ <- matrix(0, nrow = nsave, ncol = length(y))
  loglik_ <- numeric(nsave)
  sampler_params <- matrix(0, nrow = iter, ncol = 4)
  dimnames(sampler_params) <- list(NULL, c("accept_ratio__", "stepsize__", 
                                           constraint_name, "divergent__"))
  
  if (adapt_Eps) {
    Hbar <- 0
    #eps <- find_reasonable_epsilon(theta[1,], Minv, Misqrt, X, B, param)
    eps <- init_stepsize(theta, Minv, Misqrt, X, B, bnn.params)
    log_epsbar <- log(eps)
    mu <- log(10 * eps)
  }
  
  # Start of MCMC chain
  if (verbose > 0) {
    inform('')
    inform(paste('Starting', params[["sampler"]], 'at', time_start <- Sys.time()))
  }

  for (i in 1:iter) {
    if (i == 1 && verbose == 2) {
      handlers("progress")
      handlers(global = TRUE)
      pb <- progressor(warmup)
    }
    
    info <- list(0, 0)
    names(info) <- c(eval(constraint_name), "divergent__")
    
    draw <- sampling(theta, X, B, bnn.params, eps, Minv, Misqrt, constraint, info)
    theta <- draw$theta
    accept_prob <- draw$accept_prob
    WB <- theta2WB(theta, bnn.params)
    
    # Gibbs sampler for scales
    bnn.params <- 
      gibbs.update(bnn.params, 
                   WB, 
                   params[["prior"]],
                   params[["sigma.prior.a"]], 
                   params[["sigma.prior.b"]], 
                   params[["lambda.prior.a"]], 
                   params[["lambda.prior.b"]])
    
    if (i <= warmup) {
      if (verbose == 2) pb(message = "Warming up...")
      if (adapt_Eps) {
        # Dual avergaing to adapt step size
        eta <- 1 / (i + t0)
        Hbar <- (1 - eta) * Hbar + eta * (delta - accept_prob)
        log_eps <- mu - sqrt(i) * Hbar / gamma
        eta <- i^-kappa
        log_epsbar <- (1 - eta) * log_epsbar + eta * log_eps
        eps <- exp(log_eps)
      } # End of step size adaptation
      
      if (adapt_Mass) {
        # Adaptation of mass matrix
        if (adaptation_window(window_adapter)) {
          welford_num_samples <- welford_num_samples + 1
          welford_delta <- theta - welford_m
          welford_m <- welford_m + welford_delta / welford_num_samples
          if (metric == "diag") {
            welford_m2 <- welford_m2 + (theta - welford_m) * welford_delta
          } else {
            welford_m2 <- welford_m2 + tcrossprod(theta - welford_m, welford_delta)
          }
        }
        
        if (end_adaptation_window(window_adapter)) {
          window_adapter <- compute_next_window(window_adapter)
          Minv <- welford_m2 / (welford_num_samples - 1)
          if (metric == "diag")
            Minv <- (welford_num_samples / (welford_num_samples + 5)) * Minv + 
            1e-3 * (5 / (welford_num_samples + 5))
          else
            Minv <- (welford_num_samples / (welford_num_samples + 5)) * Minv + 
            1e-3 * (5 / (welford_num_samples + 5)) * diag(npar)
          
          if (!is.finite(sum(Minv))) {
            message("WARNING ", now()," Non-finite estimates in mass matrix adaptation ",
                    "-- reverting to unit metric.")
            Minv <- rep(1, len = npar)
          }
          # Update symmetric square root
          Misqrt <- if(metric == "diag") sqrt(Minv) else chol(Minv)
          # Find new reasonable eps since it can change dramatically when mass 
          # matrix updates
          eps <- init_stepsize(theta, Minv, Misqrt, X, B, bnn.params)
          mu  <- log(10 * eps)
          # Reset the running variance calculation
          welford_num_samples <- 0
          welford_m <- rep(0, len=npar)
          if (metric == "diag")
            welford_m2 <- rep(0, npar)
          else
            welford_m2 <- matrix(0, npar, npar)
        }
        window_adapter$window_counter <- window_adapter$window_counter + 1
      } # End of mass matrix adaptation
      if (i == warmup) 
        time_warmup <- difftime(Sys.time(), time_start, units='secs')
    } else {
      # Fix stepsize after warmup
      if (i == warmup + 1) {
        handlers(global = FALSE)
        if (adapt_Eps) {
          eps <- exp(log_epsbar)
          inform(paste0("Final step size = ", round(eps, 3),
                        "; after ", warmup, " warmup iterations"))
        }
        if (verbose == 2) {
          handlers(global = TRUE)
          pb <- progressor(iter - warmup)
        }
        isave <- 1 # Initialize saving index
      }
      if (verbose == 2) pb(message = "Sampling...")
      if ((i - warmup) %% thin == 0) {
        WB_[[isave]] <- WB
        loglik_mat_[isave,] <- loglik_vec(theta, X, B, bnn.params)
        loglik_[isave] <- mean(loglik_mat_[isave,])
        isave <- isave + 1
      }
    }
    sampler_params[i,] <- c(accept_prob, eps, as.numeric(info))
  } # End of MCMC loop
  suppressWarnings(waic <- waic(loglik_mat_)$estimates[3]) # Calculate WAIC
  
  
  # Print some summary info of the chain
  if (verbose > 0) {
    handlers(global = FALSE)
    ndiv <- sum(sampler_params[-(1:warmup), 'divergent__'])
    if (ndiv > 0)
      message(paste0("WARNING: There were ", ndiv, 
                     " divergent transitions after warmup"))
    
    msg <- paste0("Final acceptance ratio = ", 
                  sprintf("%.2f", mean(sampler_params[-(1:warmup), 'accept_ratio__']))) 
    
    if (adapt_Eps) msg <- paste0(msg,", and target = ", delta)
    inform(msg)
    
    time_total <- difftime(Sys.time(), time_start, units='secs')
    print_mcmc_timing(time_warmup, time_total)
  }
  output <- list(sample = WB_, loglik = loglik_, waic = waic, 
                 diagnose = sampler_params, n.knots = params[["n.knots"]],
                 activation = params[["activation"]], X = t(X), y = y)
  return(output)
}

gibbs.update <- function(params, WB, prior, global_a, global_b, local_a, local_b) {
  
  V <- params[["V"]]
  num_layers <- length(V) - 1
  W <- WB$W
  b <- WB$b
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
      aa <- global_a + V[l+1]*(V[l]+1)/2
      bb <- global_b + (sum((t(W[[l]])/lambda_W[[l]])^2)+sum((b[[l]]/lambda_b[l])^2))/2
      sigma_b[l] <- sigma_W[l] <- 1/sqrt(rgamma(1,aa,bb))
      
      # Update local scale for weights
      cc <- local_a + V[l+1]/2
      # Separate local scale for each input
      for (j in 1:V[l]) {
        dd <- local_b+sum((W[[l]][,j]/sigma_W[l])^2)/2
        lambda_W[[l]][j] <- 1/sqrt(rgamma(1,cc,dd))
      }
      # Update local scale for biases
      ff <- local_b + sum((b[[l]]/sigma_b[l])^2)/2
      lambda_b[l] <- 1/sqrt(rgamma(1,cc,ff))
    } else if (prior == "ARD" && l == 1) {
      # Update local scale for weights
      cc <- global_a + V[2]/2
      # Separate local scale for each input
      for (j in 1:V[1]) {
        dd <- global_b+sum((W[[1]][,j])^2)/2
        lambda_W[[1]][j] <- 1/sqrt(rgamma(1,cc,dd))
      }
      # Update local scale for biases
      ff <- global_b + sum((b[[1]])^2)/2
      sigma_b[1] <- 1/sqrt(rgamma(1,cc,ff))
    } else {
      # Weigths and biases have block-wise variances
      
      # Update scale for weights
      cc <- global_a + V[l+1]*V[l]/2
      dd <- global_b/H + sum(W[[l]]^2)/2
      sigma_W[l] <- 1/sqrt(rgamma(1,cc,dd))
      
      # Update scale for biases
      ee <- global_a + V[l+1]/2
      ff <- global_b + sum((b[[l]])^2)/2
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