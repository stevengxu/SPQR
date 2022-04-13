#' @title Fitting SPQR by MLE or MAP method
#'
#' @importFrom torch `%>%` torch_tensor
SPQR.ADAM <- function(X, Y, n.knots, n.hidden, activation, method, prior,
                      hyperpar, control, verbose = TRUE, seed = NULL) {

  use.GPU <- control$use.GPU
  cuda <- torch::cuda_is_available()
  if(use.GPU && !cuda){
    warning('GPU acceleration not available, using CPU')
    use.GPU <- FALSE
  } else if (!use.GPU && cuda){
    message('GPU acceleration is available through `use.GPU=TRUE`')
  }
  device <- if (use.GPU) "cuda" else "cpu"

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
  valid_indices <- sample(1:N, size = floor(N*control$valid.pct))
  train_indices <- setdiff(1:N, valid_indices)

  train_ds <- ds(train_indices)
  train_dl <- train_ds %>% torch::dataloader(batch_size=control$batch.size, shuffle=TRUE)
  valid_ds <- ds(valid_indices)
  valid_dl <- valid_ds %>% torch::dataloader(batch_size=control$batch.size, shuffle=FALSE)

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
  Bvalid <- torch_tensor(Btotal[valid_indices,], device=device)

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
      if (valid_loss < (last_valid_loss - 0.01)) {
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
      last_valid_loss <- valid_loss
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
  out <- list(model=best.model,
              loss=list(train = last_train_loss,
                        validation = last_valid_loss),
              time=time.total,
              method=method,
              config=config,
              control=control,
              X=X,
              Y=Y)
  return(out)
}
