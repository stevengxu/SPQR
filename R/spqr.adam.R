spqr.adam.train <- 
  function(params, X, y, seed = NULL, verbose = TRUE)
{
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
    
    DeepQuinn <- nn_module(
      classname = NULL,
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
        
        self$layernum <- length(n.hidden)
        self$fc <- nn_module_list()
        
        # Input Layer
        self$fc[[1]] <- mlp(varnum, n.hidden[1], dropout[1])
        
        if (self$layernum > 1) {
          # Hidden Layers
          for (i in 2:self$layernum) {
            self$fc[[i]] <- 
              mlp(n.hidden[i-1], n.hidden[i], dropout[2])
          }
        }
    
        # Output Layer
        self$output <- nn_sequential(
          nn_linear(n.hidden[self$layernum], n.knots),
          nn_softmax(2)
        )
        
      },
      
      forward = function(x) {
        for (i in 1:self$layernum) { x <- self$fc[[i]](x) }
        x %>% self$output()
      }
    )
    
    model <- DeepQuinn(ncol(X), params[["n.hidden"]], params[["n.knots"]], 
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
    Bobs <- observed[indices] #Subset the basis 
    sumprod <- torch_sum(torch_mul(predicted,Bobs),2)
    logscore <- torch_sum(-torch_log(sumprod))
    return(logscore)
  }
  
  discrepency_loss = function(observed,predicted,indices) {
    Bobs <- observed[indices] #Subset the basis 
    sumprod <- torch_sum(torch_mul(predicted,Bobs),2)
    sumlik <- 2 * torch_mean(sumprod)
    
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
