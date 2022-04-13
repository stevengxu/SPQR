#' @importFrom torch `%>%` torch_tensor nnf_linear nnf_softmax nnf_dropout nn_batch_norm1d nn_parameter distr_normal distr_gamma

.coefs <- function(W, X, activation = "tanh") {
  .W <- W$W
  .b <- W$b
  n.layers <- length(.W)
  A <- X
  for (l in 1:n.layers) {
    A <- .W[[l]]%*%A + .b[[l]]
    if (l < n.layers) {
      if (activation == "tanh")
        A <- tanh(A)
      else if (activation == "relu")
        A <- pmax(A,0)
    }
  }
  enn <- exp(A)
  p <- sweep(enn, 2, colSums(enn), '/')
  return(t(p))
}

.basis <- function(Y, K, integral = FALSE)
{
  knots <- seq(1 / (K - 2), 1 - 1 / (K - 2), length = K - 3)
  B <- splines2::mSpline(Y, knots = knots, Boundary.knots = c(0, 1),
                         intercept = TRUE, degree = 2, integral = integral)
  return(B)
}

## torch utils
nn_SPQR_MLE <- torch::nn_module(
  classname = "nn_SPQR",
  initialize = function(V, dropout, batchnorm, activation) {

    self$act <-
      switch(activation,
             `tanh`=function(...) torch::torch_tanh(...),
             `relu`=function(...) torch::torch_relu(...),
             `sigmoid`=function(...) torch::torch_sigmoid(...))

    self$batchnorm <- batchnorm
    self$dropout <- dropout
    self$layernum <- length(V)-1
    self$fc <- torch::nn_module_list()

    for (l in 1:self$layernum) self$fc[[l]] <- nn_Linear(V[l], V[l+1])
  },

  forward = function(X) {
    # input-to-hidden block
    X <- self$fc[[1]](X)
    if (self$batchnorm) X <- nn_batch_norm1d(ncol(X))(X)
    X <- self$act(X) %>% nnf_dropout(p=self$dropout[1])

    # hidden-to-hidden block
    if (self$layernum > 2) {
      for (l in 2:(self$layernum-1)) {
        X <- self$fc[[l]](X)
        if (self$batchnorm) X <- nn_batch_norm1d(ncol(X))(X)
        X <- self$act(X) %>% nnf_dropout(p=self$dropout[2])
      }
    }
    # hidden-to-output block
    X <- self$fc[[self$layernum]](X) %>% nnf_softmax(dim=2)
    return(list(output=X, logprior=torch_tensor(0)$sum()))
  }
)

nn_Linear <- torch::nn_module(
  classname = "nn_Linear",
  initialize = function(in_features, out_features) {
    self$W <- nn_parameter(torch::torch_empty(out_features,in_features))
    self$b <- nn_parameter(torch::torch_empty(out_features))

    # initialize weights and bias
    self$reset_parameters()
  },

  reset_parameters = function() {
    torch::nn_init_xavier_uniform_(self$W)
    torch::nn_init_uniform_(self$b,-0.1,0.1)
  },

  forward = function(X) {
    nnf_linear(X,self$W,self$b)
  }
)

nn_SPQR_MAP <- torch::nn_module(
  classname = "nn_SPQR",
  initialize = function(V, dropout, batchnorm, activation, prior,
                        a_tau, b_tau, a_kappa, b_kappa, device) {

    self$device <- device
    self$act <-
      switch(activation,
             `tanh`=function(...) torch::torch_tanh(...),
             `relu`=function(...) torch::torch_relu(...),
             `sigmoid`=function(...) torch::torch_sigmoid(...))
    self$batchnorm <- batchnorm
    self$dropout <- dropout
    self$layernum <- length(V)-1
    self$fc <- torch::nn_module_list()

    # Input-to-hidden Layer
    if (prior == "GP") {
      self$fc[[1]] <- nn_BayesLinear_GP(V[1], V[2], a_kappa, b_kappa, FALSE, device=device)
      self$fc[[1]]$to(device=device)
    } else if (prior == "ARD") {
      self$fc[[1]] <- nn_BayesLinear_ARD(V[1], V[2], a_kappa, b_kappa, device=device)
      self$fc[[1]]$to(device=device)
    } else {
      self$fc[[1]] <- nn_BayesLinear_GSM(V[1], V[2], a_tau, b_tau, a_kappa, b_kappa, device=device)
      self$fc[[1]]$to(device=device)
    }

    # Hidden-to-hidden and hidden-to-output Layers
    if (self$layernum > 1) {
      # Hidden Layers
      for (l in 2:self$layernum) {
        if (prior == "GSM") {
          self$fc[[l]] <- nn_BayesLinear_GSM(V[l], V[l+1], a_tau, b_tau, a_kappa, b_kappa, device=device)
          self$fc[[l]]$to(device=device)
        } else {
          self$fc[[l]] <- nn_BayesLinear_GP(V[l], V[l+1], a_kappa, b_kappa, TRUE, device=device)
          self$fc[[l]]$to(device=device)
        }
      }
    }
  },

  forward = function(X) {
    # initialize logprior
    logprior <- torch_tensor(0, device=self$device)$sum()
    # input-to-hidden block
    result = self$fc[[1]](X)
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
        if (self$batchnorm) result$output <- nn_batch_norm1d(ncol(result$output))(result$output)
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

nn_BayesLinear_ARD <- torch::nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, a_kappa, b_kappa, device) {

    self$device <- device

    self$W <- nn_parameter(torch::torch_empty(out_features,in_features))
    # log precision hyperparameter for W
    self$lkappa_W <- nn_parameter(torch::torch_ones(1,in_features))

    self$b <- nn_parameter(torch::torch_empty(out_features))
    # log precision hyperparameter for b
    self$lkappa_b <- nn_parameter(torch_tensor(1))

    # shape and rate hyperparameters for prior of kappa_b and kappa_b
    self$tpa <- nn_parameter(torch_tensor(a_kappa), requires_grad = F)
    self$tpb <- nn_parameter(torch_tensor(b_kappa), requires_grad = F)

    # initialize weights and bias
    self$reset_parameters()
  },

  reset_parameters = function() {
    torch::nn_init_xavier_uniform_(self$W)
    torch::nn_init_uniform_(self$b,-0.1,0.1)
  },

  forward = function(X) {

    kappa_W <- self$lkappa_W$exp()
    kappa_b <- self$lkappa_b$exp()
    What <- self$W$divide(kappa_W$sqrt())
    bhat <- self$b$divide(kappa_b$sqrt())

    output <- nnf_linear(X,What,bhat)

    logprior <- torch_tensor(0,device=self$device)$sum()
    logprior$add_(distr_normal(torch_tensor(0,device=self$device), torch_tensor(1,device=self$device))$log_prob(self$W)$sum())
    logprior$add_(distr_gamma(torch_tensor(self$tpa,device=self$device), torch_tensor(self$tpb,device=self$device))$log_prob(kappa_W)$sum())
    logprior$add_(distr_normal(torch_tensor(0,device=self$device), torch_tensor(1,device=self$device))$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(torch_tensor(self$tpa,device=self$device), torch_tensor(self$tpb,device=self$device))$log_prob(kappa_b)$sum())
    return(list(output=output, logprior=logprior))
  }
)

nn_BayesLinear_GP <- torch::nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, a_kappa, b_kappa,
                        scale_by_width = FALSE, device) {

    self$device = device

    self$W <- nn_parameter(torch::torch_empty(out_features,in_features))
    # log-precision hyperparameter for W
    self$lkappa_W <- nn_parameter(torch_tensor(0))

    self$b <- nn_parameter(torch::torch_empty(out_features))
    # log-precision hyperparameter for b
    self$lkappa_b <- nn_parameter(torch_tensor(0))

    # shape and rate hyperparameters for prior of kappa_W and kappa_b
    self$tpa <- nn_parameter(torch_tensor(a_kappa), requires_grad = F)
    self$tpb <- nn_parameter(torch_tensor(b_kappa), requires_grad = F)

    if (scale_by_width) {
      self$H <- nn_parameter(torch_tensor(in_features), requires_grad = F)
    } else {
      self$H <- nn_parameter(torch_tensor(1), requires_grad = F)
    }
    # initialize weights and bias
    self$reset_parameters()
  },

  reset_parameters = function() {
    torch::nn_init_xavier_uniform_(self$W)
    torch::nn_init_uniform_(self$b,-0.1,0.1)
  },

  forward = function(X) {

    kappa_W <- self$lkappa_W$exp()
    kappa_b <- self$lkappa_b$exp()
    What <- self$W$divide(kappa_W$sqrt())
    bhat <- self$b$divide(kappa_b$sqrt())

    output <- nnf_linear(X,What,bhat)

    # initialize logprior
    logprior <- torch_tensor(0, device=self$device)$sum()
    # add logprior of W ~ N(0, 1)
    logprior$add_(distr_normal(torch_tensor(0, device=self$device), torch_tensor(1, device=self$device))$log_prob(self$W)$sum())
    # add logprior of kappa_W ~ Ga(tpa,tpb)
    logprior$add_(distr_gamma(torch_tensor(self$tpa, device=self$device),
                              torch_tensor(self$tpb$divide(self$H), device=self$device))$log_prob(kappa_W)$sum())
    logprior$add_(self$lkappa_W$sum())
    # add logprior of b ~ N(0, 1)
    logprior$add_(distr_normal(torch_tensor(0, device=self$device), torch_tensor(1, device=self$device))$log_prob(self$b)$sum())
    # add logprior of kappa_b ~ Ga(tpa,tpb)
    logprior$add_(distr_gamma(torch_tensor(self$tpa, device=self$device), torch_tensor(self$tpb, device=self$device))$log_prob(kappa_b)$sum())
    logprior$add_(self$lkappa_b$sum())
    return(list(output=output, logprior=logprior))
  }
)

nn_BayesLinear_GSM <- torch::nn_module(
  classname = "nn_BayesLinear",
  initialize = function(in_features, out_features, a_tau, b_tau,
                        a_kappa, b_kappa, device) {

    self$device <- device

    # log global precision hyperparameter
    self$ltau <- nn_parameter(torch_tensor(1))

    self$W <- nn_parameter(torch::torch_empty(out_features,in_features))
    # log local precision hyperparameter for W
    self$lkappa_W <- nn_parameter(torch::torch_ones(1,in_features))

    self$b <- nn_parameter(torch::torch_empty(out_features))
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
    torch::nn_init_xavier_uniform_(self$W)
    torch::nn_init_uniform_(self$b,-0.1,0.1)
  },

  forward = function(X) {
    tau <- self$ltau$exp()
    kappa_W <- self$lkappa_W$exp()
    kappa_b <- self$lkappa_b$exp()
    What <- self$W$divide(kappa_W$sqrt()$mul(tau$sqrt()))
    bhat <- self$b$divide(kappa_b$sqrt()$mul(tau$sqrt()))

    output <- nnf_linear(X,What,bhat)

    logprior <- torch_tensor(0, device=self$device)$sum()
    logprior$add_(distr_normal(torch_tensor(0, device=self$device), torch_tensor(1, device=self$device))$log_prob(self$W)$sum())
    logprior$add_(distr_gamma(torch_tensor(self$kpa, device=self$device), torch_tensor(self$kpb, device=self$device))$log_prob(kappa_W)$sum())
    logprior$add_(distr_normal(torch_tensor(0, device=self$device), torch_tensor(1, device=self$device))$log_prob(self$b)$sum())
    logprior$add_(distr_gamma(torch_tensor(self$kpa, device=self$device), torch_tensor(self$kpb, device=self$device))$log_prob(kappa_b)$sum())
    logprior$add_(distr_gamma(torch_tensor(self$tpa, device=self$device), torch_tensor(self$tpb, device=self$device))$log_prob(tau)$sum())
    return(list(output=output, logprior=logprior))
  }
)

## Initialize parameters for window adaptation
get.nn.params <- function(fitted.obj){
    a <- fitted.obj$model$parameters
    ffnn_params <- list()
    for(j in 1:length(a)){
        ffnn_params[[j]] <- torch::as_array(a[[j]])
    }
    return(ffnn_params)
}
