SPQR.adam <- function(params, X, Y, verbose = TRUE) {
  
  V <- c(ncol(X), params$n.hidden, params$n.knots)
  # Define dataset and dataloader
  ds <- dataset(
    # modified from https://torch.mlverse.org/start/custom_dataset/  
    # See example in above link on how to set up categorical variables etc
    # I'm extracting the index variable since I need it for my custom loss function
    initialize = function(indices) {
      self$x <- torch_tensor(X[indices,,drop = FALSE])
      self$y <- torch_tensor(Y[indices])
    },
    
    .getitem = function(i) {
      list(x = self$x[i,], y = self$y[i], index = i)
    },
    
    .length = function() {
      self$y$size()[[1]]
    }
  )
  N <- nrow(X)
  valid_indices <- sample(1:N, size = floor(N*params$valid.pct))
  train_indices <- setdiff(1:N, valid_indices)
  
  train_ds <- ds(train_indices)
  train_dl <- train_ds %>% dataloader(batch_size = params$batch.size, shuffle = TRUE)
  valid_ds <- ds(valid_indices)
  valid_dl <- valid_ds %>% dataloader(batch_size = params$batch.size, shuffle = FALSE)
  
  # Use the default template
  if (params$method == "MAP") {
    model <- nn_SPQR_MAP(V, # MAP estimation using one of the three priors
                         params$dropout, 
                         params$batchnorm, 
                         params$activation, 
                         params$prior,
                         params$hyperpar$a_sigma,
                         params$hyperpar$b_sigma,
                         params$hyperpar$a_lambda,
                         params$hyperpar$b_lambda)
  } else {
    model <- nn_SPQR_MLE(V, # MLE estimation
                         params$dropout, 
                         params$batchnorm, 
                         params$activation)
  }
  
  # Computing the basis and converting it to a tensor beforehand to save
  # computational time; this is used every iteration for computing loss
  Btotal <- .basis(Y, params$n.knots)
  Btrain <- torch_tensor(Btotal[train_indices,])
  Bvalid <- torch_tensor(Btotal[valid_indices,])
  
  # Define custom loss function
  nll.loss = function(indices, basis, coefs) {
    loglik <- basis[indices,]$mul(coefs)$sum(2)$log()$sum()
    return(-loglik)
  }
  
  optimizer <- optim_adam(model$parameters, lr = params$lr)
  counter <- 0
  if(!dir.exists(params$save.path))
    dir.create(params$save.path)
  
  save_name <- file.path(params$save.path, params$save.name)
  last_valid_loss <- Inf
  last_train_loss <- Inf
  time.start <- Sys.time()
  for (epoch in 1:params$epochs) {
    
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
      if (epoch == 1 || epoch %% params$print.every.epochs == 0) {
        cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, 
                    train_loss, valid_loss))
      }
    }
    if (!is.null(params$early.stopping.epochs)) {
      if (valid_loss < (last_valid_loss - 0.01)) {
        torch_save(model, save_name)
        last_valid_loss <- valid_loss
        last_train_loss <- train_loss
        counter <- 0
      } else {
        counter <- counter + 1
        if (counter >= params$early.stopping.epochs) {
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
  time.total <- difftime(Sys.time(), time.start, units='mins')
  # load best model
  best.model <- torch_load(save_name)
  out <- list(model=best.model, 
              loss=list(train = last_train_loss,
                        validation = last_valid_loss),
              time=time.total,
              params=params,
              X=X, 
              Y=Y)
  return(out)
}


cv.SPQR.adam <- function(params, X, Y, folds, verbose) {
    
  K <- length(folds)
  V <- c(ncol(X),params$n.hidden,params$n.knots)
  # Define dataset and dataloader
  ds <- dataset(
    initialize = function(indices) {
      self$x <- torch_tensor(X[indices,,drop = FALSE])
      self$y <- torch_tensor(Y[indices])
    },
    
    .getitem = function(i) {
      list(x = self$x[i,], y = self$y[i], index = i)
    },
    
    .length = function() {
      self$y$size()[[1]]
    }
  )
  N <- nrow(X)
  # Computing the basis and converting it to a tensor beforehand to save
  # computational time; this is used every iteration for computing loss
  Btotal <- .basis(Y, params$n.knots)
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
    train_dl <- train_ds %>% dataloader(batch_size = params$batch.size, shuffle = TRUE)
    valid_ds <- ds(valid_indices)
    valid_dl <- valid_ds %>% dataloader(batch_size = params$batch.size, shuffle = FALSE)
    if (params$method == "MAP") {
      model <- nn_SPQR_MAP(V, # MAP estimation using one of the three priors
                           params$dropout, 
                           params$batchnorm, 
                           params$activation, 
                           params$prior,
                           params$hyperpar$a_sigma,
                           params$hyperpar$b_sigma,
                           params$hyperpar$a_lambda,
                           params$hyperpar$b_lambda)
    } else {
      model <- nn_SPQR_MLE(V, # MLE estimation
                           params$dropout, 
                           params$batchnorm, 
                           params$activation)
    }
    
    Btrain <- torch_tensor(Btotal[train_indices,])
    Bvalid <- torch_tensor(Btotal[valid_indices,])
    
    optimizer <- optim_adam(model$parameters, lr = params$lr)
    counter <- 0
    
    if (verbose) {
      cat(sprintf("Starting fold: %d/%d\n", k, K))
    }
    
    last_valid_loss <- Inf
    last_train_loss <- Inf
    for (epoch in 1:params$epochs) {
      
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
        if (epoch == 1 || epoch %% params$print.every.epochs == 0) {
          cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, 
                      train_loss, valid_loss))
        }
      }
      if (!is.null(params$early.stopping.epochs)) {
        if (valid_loss < (last_valid_loss - 0.01)) {
          last_valid_loss <- valid_loss
          last_train_loss <- train_loss
          counter <- 0
        } else {
          counter <- counter + 1
          if (counter >= params$early.stopping.epochs) {
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
  out <- list(params=params, cve=mean(cv_losses), folds=folds)
  class(out) <- "SPQR.adam.cv"
  invisible(out)
}

nn_SPQR_MLE <- nn_module(
  classname = "nn_SPQR",
  initialize = function(V, dropout, batchnorm, activation) {
    
    self$act <- eval(parse(text = paste0("nnf_", activation)))
    self$batchnorm <- batchnorm
    self$dropout <- dropout
    self$layernum <- length(V)-1
    self$fc <- nn_module_list()
    
    for (l in 1:self$layernum)
      self$fc[[l]] <- nn_Linear(V[l], V[l+1])
  },
  
  forward = function(X) {
    
    # input-to-hidden block
    X <- self$fc[[1]](X)
    if (self$batchnorm)
      X <- nn_batch_norm1d(ncol(X))(X)
    
    X <- self$act(X) %>% nnf_dropout(p=self$dropout[1])
    
    # hidden-to-hidden block
    if (self$layernum > 2) {
      for (l in 2:(self$layernum-1)) {
        X <- self$fc[[l]](X)
        if (self$batchnorm)
          X <- nn_batch_norm1d(ncol(X))(X)
        
        X <- self$act(X) %>% nnf_dropout(p=self$dropout[2])
      }
    }
    # hidden-to-output block
    X <- self$fc[[self$layernum]](X) %>% nnf_softmax(dim=2)
    return(list(output=X, logprior=torch_tensor(0)$sum()))
  }
)

nn_Linear <- nn_module(
  classname = "nn_Linear",
  initialize = function(in_features, out_features) {
    
    self$W <- nn_parameter(torch_empty(out_features,in_features))
    self$b <- nn_parameter(torch_empty(out_features))
  
    # initialize weights and bias
    self$reset_parameters()
  },
  
  reset_parameters = function() {
    nn_init_xavier_uniform_(self$W)
    nn_init_uniform_(self$b,-0.1,0.1)
  },
  
  forward = function(X) {
    nnf_linear(X,self$W,self$b)
  }
)

nn_SPQR_MAP <- nn_module(
  classname = "nn_SPQR",
  initialize = function(V, dropout, batchnorm, activation, prior_class, 
                        a_tau, b_tau, a_kappa, b_kappa) {
    
    self$act <- eval(parse(text = paste0("nnf_", activation)))
    self$batchnorm <- batchnorm
    self$dropout <- dropout
    self$layernum <- length(V)-1
    self$fc <- nn_module_list()
    
    # Input-to-hidden Layer
    if (prior_class == "GP") {
      self$fc[[1]] <- 
        nn_BayesLinear_GP(V[1], V[2], a_tau, b_tau, FALSE)
    } else if (prior_class == "ARD") {
      self$fc[[1]] <- 
        nn_BayesLinear_ARD(V[1], V[2], a_tau, b_tau)
    } else {
      self$fc[[1]] <- 
        nn_BayesLinear_GSM(V[1], V[2], a_tau, b_tau, a_kappa, b_kappa)
    }
    
    # Hidden-to-hidden and hidden-to-output Layers
    if (self$layernum > 1) {
      # Hidden Layers
      for (l in 2:self$layernum) {
        if (prior_class == "GSM") {
          self$fc[[l]] <- 
            nn_BayesLinear_GSM(V[l], V[l+1], a_tau, b_tau, a_kappa, b_kappa)
        } else {
          self$fc[[l]] <- 
            nn_BayesLinear_GP(V[l], V[l+1], a_tau, b_tau, TRUE)
        }
      }
    }
  },
  
  forward = function(X) {
    # initialize logprior
    logprior <- torch_tensor(0)$sum()
    # input-to-hidden block
    result <- self$fc[[1]](X)
    # accumulate logprior
    logprior$add_(result$logprior)
    # batchnorm
    if (self$batchnorm)
      result$output <- nn_batch_norm1d(ncol(result$output))(result$output)
    
    result$output <- self$act(result$output) %>% nnf_dropout(p=self$dropout[1])
    
    # hidden-to-hidden block
    if (self$layernum > 2) {
      for (l in 2:(self$layernum-1)) {
        result <- self$fc[[l]](result$output)
        logprior$add_(result$logprior)
        if (self$batchnorm)
          result$output <- nn_batch_norm1d(ncol(result$output))(result$output)
        
        result$output <- self$act(result$output) %>% nnf_dropout(p=self$dropout[2])
      }
    }
    
    # hidden-to-output block
    result <- self$fc[[self$layernum]](result$output)
    logprior$add_(result$logprior)
    result$output <- nnf_softmax(result$output, dim=2)
    return(list(output=result$output, logprior=logprior))
  }
)

nn_BayesLinear_GP <- nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, a_tau, b_tau, 
                        scale_by_width = FALSE) {
    
    self$W <- nn_parameter(torch_empty(out_features,in_features))
    # log-precision hyperparameter for W
    self$ltau_W <- nn_parameter(torch_tensor(0))
    
    self$b <- nn_parameter(torch_empty(out_features))
    # log-precision hyperparameter for b
    self$ltau_b <- nn_parameter(torch_tensor(0))
    
    # shape and rate hyperparameters for prior of tau_W and tau_b
    self$tpa <- nn_parameter(torch_tensor(a_tau), requires_grad = F)
    self$tpb <- nn_parameter(torch_tensor(b_tau), requires_grad = F)
    
    if (scale_by_width) {
      self$H <- nn_parameter(torch_tensor(in_features), requires_grad = F)
    } else {
      self$H <- nn_parameter(torch_tensor(1), requires_grad = F)
    }
    # initialize weights and bias
    self$reset_parameters()
  },
  
  reset_parameters = function() {
    nn_init_xavier_uniform_(self$W)
    nn_init_uniform_(self$b,-0.1,0.1)
  },
  
  forward = function(X) {
    
    tau_W <- self$ltau_W$exp()
    tau_b <- self$ltau_b$exp()
    What <- self$W$divide(tau_W$sqrt())
    bhat <- self$b$divide(tau_b$sqrt())
    
    output <- nnf_linear(X,What,bhat)
    
    # initialize logprior
    logprior <- torch_tensor(0)$sum()
    # add logprior of W ~ N(0, 1)
    logprior$add_(distr_normal(0,1)$log_prob(self$W)$sum())
    # add logprior of tau_W ~ Ga(tpa,tpb)
    logprior$add_(distr_gamma(self$tpa,self$tpb$divide(self$H))$log_prob(tau_W)$sum())
    logprior$add_(self$ltau_W$sum())
    # add logprior of b ~ N(0, 1)
    logprior$add_(distr_normal(0,1)$log_prob(self$b)$sum())
    # add logprior of tau_b ~ Ga(tpa,tpb)
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(tau_b)$sum())
    logprior$add_(self$ltau_b$sum())
    return(list(output=output, logprior=logprior))
  }
)

nn_BayesLinear_ARD <- nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, a_tau, b_tau) {
    
    self$W <- nn_parameter(torch_empty(out_features,in_features))
    # log precision hyperparameter for W
    self$ltau_W <- nn_parameter(torch_ones(1,in_features))
    
    self$b <- nn_parameter(torch_empty(out_features))
    # log precision hyperparameter for b
    self$ltau_b <- nn_parameter(torch_tensor(1))
    
    # shape and rate hyperparameters for prior of tau_W and tau_b
    self$tpa <- nn_parameter(torch_tensor(a_tau), requires_grad = F)
    self$tpb <- nn_parameter(torch_tensor(b_tau), requires_grad = F)
    
    # initialize weights and bias
    self$reset_parameters()
  },
  
  reset_parameters = function() {
    nn_init_xavier_uniform_(self$W)
    nn_init_uniform_(self$b,-0.1,0.1)
  },
  
  forward = function(X) {
    
    tau_W <- self$ltau_W$exp()
    tau_b <- self$ltau_b$exp()
    What <- self$W$divide(tau_W$sqrt())
    bhat <- self$b$divide(tau_b$sqrt())
    
    output <- nnf_linear(X,What,bhat)
    
    logprior <- torch_tensor(0)$sum()
    logprior$add_(distr_normal(0,1)$log_prob(self$W)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(tau_W)$sum())
    logprior$add_(distr_normal(0,1)$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(tau_b)$sum())
    return(list(output=output, logprior=logprior))
  }
)

nn_BayesLinear_GSM <- nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, a_tau, b_tau,
                        a_kappa, b_kappa) {
    
    # log global precision hyperparameter
    self$ltau <- nn_parameter(torch_tensor(1))
    
    self$W <- nn_parameter(torch_empty(out_features,in_features))
    # log local precision hyperparameter for W
    self$lkappa_W <- nn_parameter(torch_ones(1,in_features))
    
    self$b <- nn_parameter(torch_empty(out_features))
    # log local precision hyperparameter for b
    self$lkappa_b <- nn_parameter(torch_tensor(1))
    
    # shape and rate hyperparameters for prior of tau
    self$tpa <- nn_parameter(torch_tensor(a_tau), requires_grad = F)
    self$tpb <- nn_parameter(torch_tensor(b_tau), requires_grad = F)
    # shape and rate hyperparameters for prior of kappa
    self$kpa <- nn_parameter(torch_tensor(a_kappa), requires_grad = F)
    self$kpb <- nn_parameter(torch_tensor(b_kappa), requires_grad = F)
    
    # initialize weights and bias
    self$reset_parameters()
  },
  
  reset_parameters = function() {
    nn_init_xavier_uniform_(self$W)
    nn_init_uniform_(self$b,-0.1,0.1)
  },
  
  forward = function(X) {
    
    tau <- self$ltau$exp()
    kappa_W <- self$lkappa_W$exp()
    kappa_b <- self$lkappa_b$exp()
    What <- self$W$divide(kappa_W$sqrt()$mul(tau$sqrt()))
    bhat <- self$b$divide(kappa_b$sqrt()$mul(tau$sqrt()))
    
    output <- nnf_linear(X,What,bhat)
    
    logprior <- torch_tensor(0)$sum()
    logprior$add_(distr_normal(0,1)$log_prob(self$W)$sum())
    logprior$add_(distr_gamma(self$kpa,self$kpb)$log_prob(kappa_W)$sum())
    logprior$add_(distr_normal(0,1)$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(self$kpa,self$kpb)$log_prob(kappa_b)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(tau)$sum())
    return(list(output=output, logprior=logprior))
  }
)

nnf_tanh <- function(input) {
  torch_tanh(input)
}