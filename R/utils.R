check.SPQR.params <- function(params, ...) {
  if (!identical(class(params), "list"))
    stop("`params` should be a list")
  
  # merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  if (length(intersect(names(params),
                       names(dot_params))) > 0)
    stop("Same parameters in `params` and in the call are not allowed. Please check your `params` list.")
  params <- c(params, dot_params)
  
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
  
  params$method <- match.arg(params$method, c("MLE","MAP","Bayes"))
  # check shared parameters
  if (is.null(params$n.knots))
    stop("`n.knots` should be specified.")
  if (is.null(params$n.hidden))
    stop("`n.hidden` should be specified.")
  if (params$n.knots < 8)
    stop("Very small `n.knots` can lead to severe underfitting, We recommend setting it to at least 8.")
  params$activation <- match.arg(params$activation, c("tanh","relu","sigmoid")) 
  params$prior <- match.arg(params$prior, c("GP","ARD","GSM")) 
  # check method specific parameters
  if (params$method == "Bayes") {
    params <- .update.params.mcmc(params)
    params$sampler <- match.arg(params$sampler, c("HMC","NUTS"))
  } else {
    params <- .update.params.adam(params)
  }
  return(params)
}

.update.params.adam <- function(params) {
  default <- list(
    ###  shared parameters ###
    n.hidden = NULL,
    n.knots = NULL,
    activation = "tanh",
    prior = "ARD",
    hyperpar = list(a_sigma = 0.001, 
                    b_sigma = 0.001,
                    a_lambda = 0.5,
                    b_lambda = 0.5),
    #------------------------#
    # MLE and MAP parameters #
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
  if (!is.null(params)) {
    for (i in names(params))
      default[[i]] <- params[[i]]
  }
  invisible(default)
}

.update.params.mcmc <- function(params) {
  default <- list(
    ###  shared parameters ###
    n.hidden = NULL,
    n.knots = NULL,
    activation = "tanh",
    prior = "ARD",
    hyperpar = list(a_sigma = 0.001, 
                    b_sigma = 0.001,
                    a_lambda = 0.5,
                    b_lambda = 0.5),
    #------------------------#
    ###  MCMC parameters  ####
    sampler = "NUTS", 
    iter = 1000, 
    warmup = 500, 
    thin = 1,
    control = NULL
  )
  if (!is.null(params)) {
    for (i in names(params))
      default[[i]] <- params[[i]]
  }
  invisible(default)
}

.basis <- function(Y, K, integral = FALSE)
{
  knots <- seq(1 / (K - 2), 1 - 1 / (K - 2), length = K - 3)
  B <- mSpline(Y, knots = knots, Boundary.knots = c(0, 1), 
               intercept = TRUE, degree = 2, integral = integral)
  return(B)
}

## Initialize parameters for window adaptation

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


## Generate random (stratified if needed) CV folds
## Borrowed from xgboost::generate.cv.folds
SPQR.createFolds <- function(Y, nfold, stratified=FALSE) {
  # shuffle
  nrows <- length(Y)
  rnd_idx <- sample.int(nrows)
  if (stratified) {
    # stratified by quantiles
    Y <- Y[rnd_idx]
    cuts <- floor(length(Y) / nfold)
    if (cuts < 2) cuts <- 2
    if (cuts > 5) cuts <- 5
    Y <- cut(Y,
             unique(stats::quantile(Y, probs = seq(0, 1, length = cuts))),
             include.lowest = TRUE)
    
    if (nfold < length(Y)) {
      ## reset levels so that the possible levels and
      ## the levels in the vector are the same
      Y <- factor(as.character(Y))
      numInClass <- table(Y)
      foldVector <- vector(mode = "integer", length(Y))
      
      ## For each class, balance the fold allocation as far
      ## as possible, then resample the remainder.
      ## The final assignment of folds is also randomized.
      for (i in seq_along(numInClass)) {
        ## create a vector of integers from 1:nfold as many times as possible without
        ## going over the number of samples in the class. Note that if the number
        ## of samples in a class is less than nfold, nothing is produced here.
        seqVector <- rep(seq_len(nfold), numInClass[i] %/% nfold)
        ## add enough random integers to get  length(seqVector) == numInClass[i]
        if (numInClass[i] %% nfold > 0) seqVector <- c(seqVector, sample.int(nfold, numInClass[i] %% nfold))
        ## shuffle the integers for fold assignment and assign to this classes's data
        ## seqVector[sample.int(length(seqVector))] is used to handle length(seqVector) == 1
        foldVector[Y == dimnames(numInClass)$Y[i]] <- seqVector[sample.int(length(seqVector))]
      }
    } else {
      foldVector <- seq(along = Y)
    }
    
    folds <- split(seq(along = Y), foldVector)
    names(folds) <- NULL
  } else {
    # make simple non-stratified folds
    kstep <- length(rnd_idx) %/% nfold
    folds <- list()
    for (i in seq_len(nfold - 1)) {
      folds[[i]] <- rnd_idx[seq_len(kstep)]
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }
    folds[[nfold]] <- rnd_idx
  }
  return(folds)
}

get.nn.params <- function(fitted.obj){
    a <- fitted.obj$model$parameters
    ffnn_params <- list()
    for(j in 1:length(a)){
        ffnn_params[[j]] <- as_array(a[[j]])  
    }
    return(ffnn_params)
}