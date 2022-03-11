#ifndef BNN_MCMC_H
#define BNN_MCMC_H

#include "utils.h"
#include "activation.h"
using std::string;

// [[Rcpp::export]]
Rcpp::NumericVector loglik_vec(
    const arma::vec& theta, // posterior sample
    const arma::mat& X, // data matrix
    const arma::mat& B, // basis matrix
    const Rcpp::List& param // model parameters
)
{
  arma::vec V = param["V"]; // number of hidden neurons
  unsigned int num_layers = V.n_elem - 1;
  string activation = param["activation"];
  
  // Weight matrices
  arma::mat W[num_layers];
  // Bias vectors
  arma::vec b[num_layers];
  // Placeholder for forward propagation, initialized to X
  arma::mat A(X);
  
  // Read in parameter values, accumulate log-prior, and forward prop
  unsigned int ii = 0;
  for (unsigned int l = 0; l < num_layers; l++) {
    W[l] = arma::mat(V[l+1],V[l]);
    b[l] = arma::vec(V[l+1]);
    for (unsigned int i = 0; i < W[l].n_elem; i++) {
      W[l](i) = theta(ii++);
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      b[l](i) = theta(ii++);
    }
    A = W[l] * A;
    A.each_col() += b[l];
    if (l == num_layers - 1) {
      // Output activation
      A = colSoftmax(A);
    } else {
      // Hidden activation
      if (activation == "tanh") {
        A = tanh(A);
      } else {
        A = relu(A); 
      }
    }
  }  
  arma::vec ll_vec = log(sum(B % A, 0)).t();
  return arma2rvec(ll_vec);
}

// [[Rcpp::export]]
double logprob(
    const arma::vec& theta, // posterior sample
    const arma::mat& X, // data matrix
    const arma::mat& B, // basis matrix
    const Rcpp::List& param // model parameters
)
{
  arma::vec V = param["V"]; // number of hidden neurons
  unsigned int num_layers = V.n_elem - 1;
  string activation = param["activation"];
  Rcpp::List lambda_W_ = param["lambda_W"];
  
  double logprior = 0.0;  
  double loglik = 0.0;
  
  // Weight matrices
  arma::mat W[num_layers];
  // Bias vectors
  arma::vec b[num_layers];
  // Placeholder for forward propagation, initialized to X
  arma::mat A(X);
  // Scale parameters for weights and bias
  // 
  // W^(l)_ij ~ N(0, sigma[l]*lambda_W[l](j)
  // b^(l)_i ~ N(0, sigma[l]*lambda_b[l])
  arma::vec sigma_W = param["sigma_W"]; // Global scale for weights
  arma::vec sigma_b = param["sigma_b"]; // Scale for bias
  arma::vec lambda_W[num_layers];
  arma::vec lambda_b = param["lambda_b"]; // feature-wise scale
  lambda_b %= sigma_b;
  
  // Read in parameter values and 
  // accumulate log-prior
  unsigned int ii = 0;
  for (unsigned int l = 0; l < num_layers; l++) {
    lambda_W[l] = Rcpp::as<std::vector<double>>(lambda_W_[l]);
    lambda_W[l] *= sigma_W[l];
    W[l] = arma::mat(V[l+1],V[l]);
    b[l] = arma::vec(V[l+1]);
    for (unsigned int j = 0; j < W[l].n_cols; j++) {
      for (unsigned int i = 0; i < W[l].n_rows; i++) {
        W[l](i,j) = theta(ii++);
        logprior += R::dnorm(W[l](i,j), 0.0, lambda_W[l](j), true);
      }
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      b[l](i) = theta(ii++);
      logprior += R::dnorm(b[l](i), 0.0, lambda_b[l], true);
    }
    A = W[l] * A;
    A.each_col() += b[l];
    if (l == num_layers - 1) {
      // Output activation
      A = colSoftmax(A);
    } else {
      if (activation == "tanh") {
        A = tanh(A);
      } else {
        A = relu(A); 
      }
    }
  }
  loglik += sum(log(sum(B % A, 0)));

  return logprior + loglik;
}

// [[Rcpp::export]]
arma::vec glogprob(
    const arma::vec& theta, // posterior sample
    const arma::mat& X, // data matrix
    const arma::mat& B, // basis matrix
    const Rcpp::List& param // model parameters
)
{
  arma::vec V = param["V"]; // number of hidden neurons
  unsigned int num_layers = V.n_elem - 1;
  string activation = param["activation"];
  Rcpp::List lambda_W_ = param["lambda_W"];
  
  // Weight matrices
  arma::mat W[num_layers];
  // Bias vectors
  arma::vec b[num_layers];
  // Placeholders for forward propagation
  arma::mat A[num_layers];
  A[0] = X;
  arma::mat Z;
  // Scale parameters for weights and bias
  arma::vec sigma_W = param["sigma_W"]; // Global scale for weights
  arma::vec sigma_b = param["sigma_b"]; // Scale for bias
  arma::vec lambda_W[num_layers];
  arma::vec lambda_b = param["lambda_b"]; // feature-wise scale
  lambda_b %= sigma_b;
  
  // Containers for gradients
  arma::mat grad_W[num_layers];
  arma::vec grad_b[num_layers];
  
  unsigned int ii = 0;
  for (unsigned int l = 0; l < num_layers; l++) {
    lambda_W[l] = Rcpp::as<std::vector<double>>(lambda_W_[l]);
    lambda_W[l] *= sigma_W[l];
    W[l] = arma::mat(V[l+1],V[l]);
    b[l] = arma::vec(V[l+1]);
    grad_W[l] = arma::mat(V[l+1],V[l]);
    grad_b[l] = arma::vec(V[l+1]);
    for (unsigned int j = 0; j < W[l].n_cols; j++) {
      for (unsigned int i = 0; i < W[l].n_rows; i++) {
        W[l](i,j) = theta(ii++);
        grad_W[l](i,j) = -W[l](i,j) / pow(lambda_W[l](j),2);
      }
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      b[l](i) = theta(ii++);
      grad_b[l](i) = -b[l](i) / pow(lambda_b[l],2);
    }
    Z = W[l] * A[l];
    Z.each_col() += b[l];
    if (l < num_layers - 1) {
      if (activation == "tanh") {
        A[l+1] = tanh(Z);
      } else {
        A[l+1] = relu(Z);
      }
    }
  }
  
  // Softmax transformation
  arma::rowvec maxv(Z.n_cols);
  for (unsigned int i = 0; i < maxv.n_elem; i++) {
    unsigned int maxi = Z.col(i).index_max();
    maxv(i) = Z(maxi, i);
  }
  arma::mat enn = exp(Z.each_row() - maxv);
  arma::mat snn = B % enn;
  enn.each_row() /= sum(enn, 0);
  snn.each_row() /= sum(snn, 0);
  
  // Placeholder for back propagation
  arma::mat delta[num_layers];
  for (unsigned int l = 0; l < num_layers; l++) {
    unsigned int ll = num_layers - l - 1;
    if (ll == num_layers - 1) {
      delta[ll] = snn - enn;
    } else {
      if (activation == "tanh") {
        delta[ll] = W[ll+1].t() * delta[ll+1] % (1 - arma::square(A[ll+1]));
      } else {
        delta[ll] = W[ll+1].t() * delta[ll+1] % relu_grad(A[ll+1]);
      }
    }
    grad_W[ll] += delta[ll] * A[ll].t();
    grad_b[ll] += sum(delta[ll],1);
    grad_W[ll].reshape(grad_W[ll].n_elem,1);
  }
  
  // prepare output
  arma::vec grad(theta.n_elem);
  ii = 0;
  // d_WH
  for (unsigned int l = 0; l < num_layers; l++) {
    for (unsigned int i = 0; i < W[l].n_elem; i++) {
      grad(ii++) = grad_W[l](i);
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      grad(ii++) = grad_b[l](i);
    }
  }
  return grad;
}

#endif