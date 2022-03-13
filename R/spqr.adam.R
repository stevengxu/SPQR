spqr.adam.train <- 
  function(params, X, y, seed = NULL, verbose = TRUE) {
    
  if (is.null(dim(X))) X <- as.matrix(X)
  # Define dataset and dataloader
  quinn_dataset <- dataset(
    # modified from https://torch.mlverse.org/start/custom_dataset/  
    # See example in above link on how to set up categorical variables etc
    # I'm extracting the index variable since I need it for my custom loss function
    name = "train",
    
    initialize = function(indices) {
      #df <- na.omit(df) 
      self$x <- torch_tensor(as.matrix(X[indices,]))
      self$y <- torch_tensor(y[indices])
    },
    
    .getitem = function(i) {
      list(x = self$x[i,], y = self$y[i], index = i)
    },
    
    .length = function() {
      self$y$size()[[1]]
    }
  )
  
  train_indices <- sample(1:nrow(X), size = floor(0.8 * nrow(X)))
  valid_indices <- setdiff(1:nrow(X), train_indices)
  
  train_ds <- quinn_dataset(train_indices)
  train_dl <- train_ds %>% dataloader(batch_size = params[["batch.size"]], 
                                      shuffle = TRUE)
  valid_ds <- quinn_dataset(valid_indices)
  valid_dl <- valid_ds %>% dataloader(batch_size = params[["batch.size"]], 
                                      shuffle = FALSE)
  
  if (is.null(params[["model"]])) {
    # Use the default template
    
    nn_spqr <- nn_module(
      classname = "nn_spqr",
      initialize = function(varnum, n.hidden = c(8,8), n.knots, 
                            dropout = c(0,0), batchnorm = F, 
                            activation = "tanh") {
        
        hiddenAct <- eval(parse(text = paste0("nn_", activation)))
        if (batchnorm) {
          mlp <- function(in_features, out_features, probs) {
            nn_sequential(
              nn_linear(in_features, out_features),
              nn_batch_norm1d(out_features),
              hiddenAct(),
              nn_dropout(probs))
          }
        } else {
          mlp <- function(in_features, out_features, probs) {
            nn_sequential(
              nn_linear(in_features, out_features),
              hiddenAct(),
              nn_dropout(probs))
          }
        }
        
        self$layernum <- length(n.hidden)+1
        self$fc <- nn_module_list()
        
        # Input Layer
        self$fc[[1]] <- mlp(varnum, n.hidden[1], dropout[1])
        
        if (self$layernum > 2) {
          # Hidden Layers
          for (i in 2:(self$layernum-1)) {
            self$fc[[i]] <- 
              mlp(n.hidden[i-1], n.hidden[i], dropout[2])
          }
        }
        
        # Output Layer
        self$fc[[self$layernum]] <- nn_sequential(
          nn_linear(n.hidden[self$layernum-1], n.knots),
          nn_softmax(2)
        )
        
      },
      
      forward = function(x) {
        for (i in 1:self$layernum) { x <- self$fc[[i]](x) }
        x
      }
    )
    
    model <- nn_spqr(ncol(X), params[["n.hidden"]], params[["n.knots"]], 
                       params[["dropout"]], params[["batchnorm"]], 
                       params[["activation"]])
    for (i in 1:model$layernum)
      nn_init_xavier_uniform_(model$fc[[i]][[1]]$weight)
    
  }
  
  # Computing the basis and converting it to a tensor beforehand to save
  # computational time; this is used every iteration for computing loss
  Btotal <- t(sp.basis(y, params[["n.knots"]]))
  Btrain <- Btotal[train_indices,]
  Bvalid <- Btotal[valid_indices,]
  Btrain_tensor <- torch_tensor(Btrain)
  Bvalid_tensor <- torch_tensor(Bvalid)
  
  # Define custom loss function
  nloglik_loss = function(observed, predicted, indices) {
    B <- observed[indices] #Subset the basis 
    probs <- B$mul(predicted)$sum(2)
    loglik <- probs$log()$sum()
    return(-loglik)
  }
  
  optimizer <- optim_adam(model$parameters, lr = params[["lr"]])
  best_loss <- Inf
  counter <- 0
  if(!dir.exists(params[["save.path"]]))
    dir.create(params[["save.path"]])
  
  save_name <- file.path(params[["save.path"]], params[["save.name"]])
  for (epoch in 1:params[["epochs"]]) {
    
    model$train()
    train_losses <- c()  
    
    coro::loop(for (b in train_dl) {
      
      optimizer$zero_grad()
      output <- model(b$x)
      indices <- b$index
      loss <- nloglik_loss(observed = Btrain_tensor, predicted = output, 
                           indices = indices)
      
      loss$backward()
      optimizer$step()
      
      train_losses <- c(train_losses, loss$item())
      
    })
    
    model$eval()
    valid_losses <- c()
    
    coro::loop(for (b in valid_dl) {
      
      output <- model(b$x)
      indices <- b$index
      loss <- nloglik_loss(observed = Bvalid_tensor, predicted = output, 
                           indices = indices)
      
      valid_losses <- c(valid_losses, loss$item())
      
    })
    train_loss <- mean(train_losses)
    valid_loss <- mean(valid_losses)
    if (verbose)
      cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, 
                  train_loss, valid_loss))
    if (valid_loss < (best_loss - 0.01)) {
      torch_save(model, save_name)
      best_loss <- valid_loss
      counter <- 0
    } else {
      counter <- counter + 1
      if (counter >= params[["patience"]]) { break }
    }
  }
  
  best_model <- torch_load(save_name)
  out <- list(model = best_model, loglik = -best_loss, X = X, y = y, 
              n.knots = n.knots)
  return(out)
}

# fully connect NN block with GP prior

nn_spqr_GP <- nn_module(
  classname = "nn_spqr",
  initialize = function(V, dropout, batchnorm, activation, tau_prior_a, tau_prior_b) {
    
    activation <- eval(parse(text = paste0("nn_", activation)))
    if (batchnorm) {
      fc_block <- nn_BayesFC_GP_batchnorm
    } else {
      fc_block <- nn_BayesFC_GP
    }
    
    self$layernum <- length(V)-1
    self$fc <- nn_module_list()
    
    # Input Layer
    self$fc[[1]] <- fc_block(V[1], V[2], dropout[1], tau_prior_a, 
                             tau_prior_b, FALSE)
    
    if (self$layernum > 2) {
      # Hidden Layers
      for (i in 2:(self$layernum-1)) {
        self$fc[[i]] <- 
          fc_block(V[i], V[i+1], dropout[2], tau_prior_a, 
                   tau_prior_b, TRUE)
      }
    }
    
    # Output Layer
    self$fc[[self$layernum]] <- nn_sequential(
      nn_BayesLinear_GP(V[self$layernum], V[self$layernum+1]),
      nn_softmax(2)
    )
    
  },
  
  forward = function(X) {
    
    logprior <- torch_tensor(0)$sum()
    result <- self$fc[[1]](X)
    logprior$add_(result$logprior)
    for (i in 2:self$layernum) {
      result <- self$fc[[i]](result$output)
      logprior$add_(result$logprior)
    }
    return(list(output=result$output, logprior=logprior))
  }
)

nn_BayesFC_GP_batchnorm <- function(in_features, out_features, probs,
                                     tau_prior_a, tau_prior_b, scale_by_width) {
  nn_sequential(
    nn_BayesLinear_GP(in_features, out_features, tau_prior_a, 
                      tau_prior_b, scale_by_width),
    nn_batch_norm1d(out_features),
    activation(),
    nn_dropout(probs))
}

nn_BayesFC_GP <- function(in_features, out_features, probs,
                          tau_prior_a, tau_prior_b, scale_by_width) {
  nn_sequential(
    nn_BayesLinear_GP(in_features, out_features, tau_prior_a, 
                      tau_prior_b, scale_by_width),
    activation(),
    nn_dropout(probs))
}

nn_BayesLinear_GP <- nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, tau_prior_a, tau_prior_b, scale_by_width = FALSE) {
    
    self$W <- nn_parameter(nn_init_xavier_uniform_(torch_zeros(out_features,in_features)))
    # precision hyperparameter for W
    self$tau_W <- nn_parameter(torch_tensor(1))
    
    self$b <- nn_parameter(nn_init_xavier_uniform_(torch_zeros(out_features,1)))
    # precision hyperparameter for b
    self$tau_b <- nn_parameter(torch_tensor(1))
    
    # shape and rate hyperparameters for prior of tau_W and tau_b
    self$tpa <- torch_tensor(tau_prior_a)
    self$tpb <- torch_tensor(tau_prior_b)
    
    if (scale_by_width) {
      self$H <- torch_tensor(in_features)
    } else {
      self$H <- torch_tensor(1)
    }
  },
  
  forward = function(X) {
    output <- torch_mm(self$W,X)$add(self$b)
    sig_W <- torch_divide(1,self$tau_W$sqrt())
    sig_b <- torch_divide(1,self$tau_b$sqrt())
    logprior <- torch_tensor(0)$sum()
    logprior$add_(distr_normal(0,sig_W)$log_prob(self$W)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb$divide(self$H))$log_prob(self$tau_W)$sum())
    logprior$add_(distr_normal(0,sig_b)$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(self$tau_b)$sum())
    return(list(output=output, logprior=logprior))
  }
)

# fully connect NN block with ARD prior

nn_spqr_ARD <- nn_module(
  classname = "nn_spqr",
  initialize = function(V, dropout, batchnorm, activation, tau_prior_a, tau_prior_b) {
    
    activation <- eval(parse(text = paste0("nn_", activation)))
    fc_block <- nn_module_list()
    if (batchnorm) {
      fc_block[[1]] <- nn_BayesFC_ARD_batchnorm
      fc_block[[2]] <- nn_BayesFC_GP_batchnorm
    } else {
      fc_block[[1]] <- nn_BayesFC_ARD
      fc_block[[2]] <- nn_BayesFC_GP
    }
    
    self$layernum <- length(V)-1
    self$fc <- nn_module_list()
    
    # Input Layer
    self$fc[[1]] <- fc_block[[1]](V[1], V[2], dropout[1], tau_prior_a, 
                                  tau_prior_b, FALSE)
    
    if (self$layernum > 2) {
      # Hidden Layers
      for (i in 2:(self$layernum-1)) {
        self$fc[[i]] <- 
          fc_block[[2]](V[i], V[i+1], dropout[2], tau_prior_a, 
                        tau_prior_b, TRUE)
      }
    }
    
    # Output Layer
    self$fc[[self$layernum]] <- nn_sequential(
      nn_BayesLinear_GP(V[self$layernum], V[self$layernum+1]),
      nn_softmax(2)
    )
    
  },
  
  forward = function(X) {
    
    logprior <- torch_tensor(0)$sum()
    result <- self$fc[[1]](X)
    logprior$add_(result$logprior)
    for (i in 2:self$layernum) {
      result <- self$fc[[i]](result$output)
      logprior$add_(result$logprior)
    }
    return(list(output=result$output, logprior=logprior))
  }
)

nn_BayesFC_ARD_batchnorm <- function(in_features, out_features, probs, 
                                    tau_prior_a, tau_prior_b) {
  nn_sequential(
    nn_BayesLinear_ARD(in_features, out_features, tau_prior_a, 
                       tau_prior_b),
    nn_batch_norm1d(out_features),
    activation(),
    nn_dropout(probs))
}

nn_BayesFC_ARD <- function(in_features, out_features, probs, 
                                     tau_prior_a, tau_prior_b) {
  nn_sequential(
    nn_BayesLinear_ARD(in_features, out_features, tau_prior_a, 
                       tau_prior_b),
    activation(),
    nn_dropout(probs))
}

nn_BayesLinear_ARD <- nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, tau_prior_a, tau_prior_b) {
    
    self$W <- nn_parameter(nn_init_xavier_uniform_(torch_zeros(out_features,in_features)))
    # precision hyperparameter for W
    self$tau_W <- nn_parameter(torch_tensor(in_features))
    
    self$b <- nn_parameter(nn_init_xavier_uniform_(torch_zeros(out_features,1)))
    # precision hyperparameter for b
    self$tau_b <- nn_parameter(torch_tensor(1))
    
    # shape and rate hyperparameters for prior of tau_W and tau_b
    self$tpa <- torch_tensor(sigma_prior_a)
    self$tpb <- torch_tensor(sigma_prior_b)
    
  },
  
  forward = function(X) {
    output <- torch_mm(self$W,X)$add(self$b)
    sig_W <- torch_divide(1,self$tau_W$sqrt())
    sig_b <- torch_divide(1,self$tau_b$sqrt())
    logprior <- torch_tensor(0)$sum()
    for (i in 1:self$W$size[2]) {
      logprior$add_(distr_normal(0,sig_W[i])$log_prob(self$W[,i])$sum())
      logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(self$tau_W[i])$sum())
    }
    logprior$add_(distr_normal(0,sig_b)$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(self$tau_b)$sum())
    return(list(output=output, logprior=logprior))
  }
)

# fully connect NN block with GSM prior

nn_spqr_GSM <- nn_module(
  classname = "nn_spqr",
  initialize = function(V, dropout, batchnorm, activation, tau_prior_a, tau_prior_b, 
                        kappa_prior_a, kappa_prior_b) {
    
    activation <- eval(parse(text = paste0("nn_", activation)))
    if (batchnorm) {
      fc_block <- nn_BayesFC_GSM_batchnorm
    } else {
      fc_block <- nn_BayesFC_GSM
    }
    
    self$layernum <- length(V)-1
    self$fc <- nn_module_list()
    
    # Input Layer
    self$fc[[1]] <- fc_block(V[1], V[2], dropout[1], tau_prior_a, 
                             tau_prior_b, kappa_prior_a, kappa_prior_b, FALSE)
    
    if (self$layernum > 2) {
      # Hidden Layers
      for (i in 2:(self$layernum-1)) {
        self$fc[[i]] <- 
          fc_block(V[i], V[i+1], dropout[2], tau_prior_a, 
                   tau_prior_b, kappa_prior_a, kappa_prior_b, TRUE)
      }
    }
    
    # Output Layer
    self$fc[[self$layernum]] <- nn_sequential(
      nn_BayesLinear_GP(V[self$layernum], V[self$layernum+1]),
      nn_softmax(2)
    )
    
  },
  
  forward = function(X) {
    
    logprior <- torch_tensor(0)$sum()
    result <- self$fc[[1]](X)
    logprior$add_(result$logprior)
    for (i in 2:self$layernum) {
      result <- self$fc[[i]](result$output)
      logprior$add_(result$logprior)
    }
    return(list(output=result$output, logprior=logprior))
  }
)

nn_BayesFC_GSM_batchnorm <- function(in_features, out_features, probs, 
                                     tau_prior_a, tau_prior_b,
                                     kappa_prior_a, kappa_prior_b) {
  nn_sequential(
    nn_BayesLinear_GSM(in_features, out_features, tau_prior_a, 
                       tau_prior_b, kappa_prior_a, kappa_prior_b),
    nn_batch_norm1d(out_features),
    activation(),
    nn_dropout(probs))
}

nn_BayesFC_GSM <- function(in_features, out_features, probs, 
                           tau_prior_a, tau_prior_b,
                           kappa_prior_a, kappa_prior_b) {
  nn_sequential(
    nn_BayesLinear_GSM(in_features, out_features, tau_prior_a, 
                       tau_prior_b, kappa_prior_a, kappa_prior_b),
    activation(),
    nn_dropout(probs))
}

nn_BayesLinear_GSM <- nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, tau_prior_a, tau_prior_b,
                        kappa_prior_a, kappa_prior_b) {
    
    # global precision hyperparameter
    self$tau <- nn_parameter(torch_tensor(1))
    
    self$W <- nn_parameter(nn_init_xavier_uniform_(torch_zeros(out_features,in_features)))
    # local precision hyperparameter for W
    self$kappa_W <- nn_parameter(torch_tensor(in_features))
    
    self$b <- nn_parameter(nn_init_xavier_uniform_(torch_zeros(out_features,1)))
    # local precision hyperparameter for b
    self$kappa_b <- nn_parameter(torch_tensor(1))
    
    # shape and rate hyperparameters for prior of tau
    self$tpa <- torch_tensor(tau_prior_a)
    self$tpb <- torch_tensor(tau_prior_b)
    # shape and rate hyperparameters for prior of kappa
    self$kpa <- torch_tensor(kappa_prior_a)
    self$kpb <- torch_tensor(kappa_prior_b)
    
  },
  
  forward = function(X) {
    output <- torch_mm(self$W,X)$add(self$b)
    sig <- torch_divide(1,self$tau$sqrt())
    lambda_W <- torch_divide(1,self$kappa_W$sqrt())
    lambda_b <- torch_divide(1,self$kappa_b$sqrt())
    logprior <- torch_tensor(0)$sum()
    for (i in 1:self$W$size[2]) {
      logprior$add_(distr_normal(0,torch_mul(sig,lambda_W[i]))$log_prob(self$W[,i])$sum())
      logprior$add_(distr_gamma(self$kpa,self$kpb)$log_prob(self$kappa_W[i])$sum())
    }
    logprior$add_(distr_normal(0,torch_mul(sig,lambda_b))$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(self$kpa,self$kpb)$log_prob(self$kappa_b)$sum())
    logprior$add_(distr_gamma(self$tpa,self$tpb)$log_prob(self$tau)$sum())
    return(list(output=output, logprior=logprior))
  }
)