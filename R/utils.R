`%notin%`<- Negate(`%in%`)

check.spqr.params <- function(params, ...) {
  if (!identical(class(params), "list"))
    stop("params must be a list")
  
  # merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  if (length(intersect(names(params),
                       names(dot_params))) > 0)
    stop("Same parameters in 'params' and in the call are not allowed. Please check your 'params' list.")
  params <- c(params, dot_params)
  
  # providing a parameter multiple times makes sense only for 'eval_metric'
  name_freqs <- table(names(params))
  multi_names <- names(name_freqs[name_freqs > 1])
  if (length(multi_names) > 0) {
    warning("The following parameters were provided multiple times:\n\t",
            paste(multi_names, collapse = ', '), "\n  Only the last value for each of them will be used.\n")
    for (n in multi_names) {
      del_idx <- which(n == names(params))
      del_idx <- del_idx[-length(del_idx)]
      params[[del_idx]] <- NULL
    }
  }
  
  # check if parameter values are sensible
  if (params[["n.knots"]] < 8)
    warning("Very few knots can lead to severe underfitting, 
            We recommend setting it to at least 8.")
  
  if (params[["activation"]] %notin% c("tanh","relu"))
    stop("'activation' must be either 'tanh' or 'relu'.")
  
  if (params[["method"]] == "Bayes") {
    params <- update.mcmc.params(params)
    
    if (params[["sampler"]] %notin% c("HMC", "NUTS"))
      stop("'sampler' must be either 'HMC' or 'NUTS'.")
  
    if (params[["prior"]] %notin% c("ISO","ARD","GSM"))
      stop("'prior' must be one of c('ISO','ARD','GSM').")
    
  } else {
    params <- update.adam.params(params)
  }
  
  return(params)
}

update.adam.params <- function(params) {
  default <- list(
    n.hidden = NULL,
    n.knots = NULL,
    lr = 0.05,
    dropout = c(0,0),
    batchnorm = FALSE,
    epochs = 50,
    batch.size = NULL,
    model = NULL,
    save.path = file.path(getwd(),"spqr_model"),
    save.name = "spqr.model.pt",
    patience = 10
  )
  if (!is.null(params)) {
    for (i in names(params))
      default[[i]] <- params[[i]]
  }
  invisible(default)
}

update.mcmc.params <- function(params) {
  default <- list(
    n.hidden = NULL,
    n.knots = NULL,
    sampler = "NUTS", 
    prior = "ARD",
    iter = 1000, 
    warmup = 500, 
    thin = 1,
    sigma.prior.a = 0.001, 
    sigma.prior.b = 0.001, 
    lambda.prior.a = 0.5, 
    lambda.prior.b = 0.5,
    control = NULL
  )
  if (!is.null(params)) {
    for (i in names(params))
      default[[i]] <- params[[i]]
  }
  invisible(default)
}

## Update the control list

update.hmc.control <- function(control) {
  default <- list(
    adapt_gamma = 0.05,
    adapt_delta = 0.9,
    adapt_kappa = 0.75,
    adapt_t0 = 10,
    adapt_init_buffer = 75,
    adapt_term_buffer = 50,
    adapt_base_window = 25,
    stepsize = NULL,
    metric = "diag",
    max_tree_depth = 6,
    int_time = 0.3
  )
  if (!is.null(control)) {
    for (i in names(control))
      default[[i]] <- control[[i]]
  }
  invisible(default)
}

sp.basis <- function(y, K, integral = FALSE)
{
  knots <- seq(1 / (K - 2), 1 - 1 / (K - 2), length = K - 3)
  B <- mSpline(y, knots = knots, Boundary.knots = c(0, 1), 
               intercept = TRUE, degree = 2, integral = integral)
  return(t(B))
}

sp.coefs <- function(W, b, X, activation = "tanh")
{
  num_layers <- length(W)
  A <- X
  for (l in 1:num_layers) {
    A <- W[[l]]%*%A + b[[l]]
    if (l < num_layers) {
      if (activation == "tanh")
        A <- base::tanh(A)
      else
        A <- pmax(A,0)
    }
  }
  enn <- exp(A)
  p <- sweep(enn, 2, colSums(enn), '/')
  return(p)
}

nn.init.xavier.uniform <- function(V) {
  num_layers <- length(V) - 1
  theta <- {}
  for (l in 1:num_layers) {
    W <- sqrt(6)/sqrt(V[l]+1+V[l+1])*runif(V[l]*V[l+1],-1,1)
    b <- sqrt(6)/sqrt(V[l]+1+V[l+1])*runif(V[l+1],-1,1)
    theta <- c(theta, c(W,b))
  }
  return(theta)
}

## Extract posterior samples of weights and bias

theta2WB <- function(theta, param) {
  
  V <- param[["V"]]
  num_layers <- length(V) - 1
  
  W <- vector("list", num_layers)
  b <- vector("list", num_layers)
  
  end <- 0
  for (l in 1:num_layers) {
    start <- end + 1
    end <- start + V[l]*V[l+1] - 1
    W[[l]] <- theta[start:end]
    dim(W[[l]]) <- c(V[l+1],V[l])
    start <- end + 1
    end <- start + V[l+1] - 1
    b[[l]] <- theta[start:end]
  }
  
  return(list(W=W, b=b))
}

## Initialize parameters for window adaptation

initialize_window_adapter <- function(num_warmup, control) {
  init_buffer <- control$adapt_init_buffer
  term_buffer <- control$adapt_term_buffer
  base_window <- control$adapt_base_window
  
  if (num_warmup < (init_buffer + term_buffer + base_window)) {
    message("WARNING ", now(), " There aren't enough warmup iterations to fit the ",
            "three stages of adaptation as currently configured.")
    init_buffer <- 0.15 * num_warmup
    term_buffer <- 0.1 * num_warump
    base_window <- num_warmup - (init_buffer + term_buffer)
    inform(c(paste("Reducing each adaptation stage to 15%/75%/10% of",
                   "the given number of warmup iterations:"),
             paste("init_buffer =", init_buffer),
             paste("base_window =", base_window),
             paste("term_buffer =", term_buffer)))
  }
  adapter <- list(
    num_warmup = num_warmup,
    window_counter = 0,
    init_buffer = init_buffer,
    term_buffer = term_buffer,
    base_window = base_window,
    window_size = base_window,
    next_window = init_buffer + term_buffer
  )
  invisible(adapter)
}

## Check if iteration is within adaptation window

adaptation_window <- function(adapter) {
  c1 <- adapter$window_counter >= adapter$init_buffer
  c2 <- adapter$window_counter <= adapter$num_warmup - adapter$term_buffer
  c3 <- adapter$window_counter < adapter$num_warmup
  return(c1 && c2 && c3)
}

## Check if iteration is at the end of adaptation window

end_adaptation_window <- function(adapter) {
  c1 <- adapter$window_counter == adapter$next_window
  c2 <- adapter$window_counter < adapter$num_warmup
  return(c1 && c2)
}

## Compute the next window size in mass matrix adaptation

compute_next_window <- function(adapter) {
  
  next_window <- adapter$next_window
  num_warmup <- adapter$num_warmup
  term_buffer <- adapter$term_buffer
  window_size <- adapter$window_size
  
  if(next_window == (num_warmup - term_buffer)) 
    return(adapter)
  
  window_size <- window_size * 2
  adapter$window_size <- window_size
  next_window <- adapter$window_counter + window_size
  
  if(next_window == (num_warmup - term_buffer)) {
    adapter$next_window <- next_window
    return(adapter)
  }
  
  next_window_boundary <- next_window + 2 * window_size
  if(next_window_boundary >= num_warmup - term_buffer)
    next_window <- num_warmup - term_buffer
  
  adapter$next_window <- next_window
  invisible(adapter)
}

## Print MCMC progress to console.

print_mcmc_progress <- function(iteration, iter, warmup){
  i <- iteration
  refresh <- max(10, floor(iter / 10))
  if (i == 1 | i == iter | i %% refresh == 0){
    i.width <- formatC(i, width = nchar(iter))
    msg <- paste0('Iteration: ', i.width , "/", iter, " [",
                  formatC(floor(100 * (i / iter)), width = 3), "%]",
                  ifelse(i <= warmup, " (Warmup)", " (Sampling)"))
    inform(msg)
  }
}

## Print MCMC timing to console

print_mcmc_timing <- function(time_warmup, time_total){
  x <- ' Elapsed Time: '
  inform(paste0(x, sprintf("%.1f", time_warmup), ' seconds (Warmup)'))
  inform(paste0(x, sprintf("%.1f", time_total-time_warmup), ' seconds (Sampling)'))
  inform(paste0(x, sprintf("%.1f", time_total), ' seconds (Total)'))
}