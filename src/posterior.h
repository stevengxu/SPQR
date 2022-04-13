#ifndef POSTERIOR_H
#define POSTERIOR_H

#include "utils.h"
#include "activation.h"
using std::string;

// [[Rcpp::export]]
Rcpp::NumericVector loglik_vec
  (
    const arma::vec& theta,
    const arma::mat& X,
    const arma::mat& B,
    const Rcpp::List& param
  )
{
  arma::vec V = param["V"]; // number of hidden neurons
  const unsigned int num_layers = V.n_elem - 1;
  string activation = param["activation"];

  // Weight matrices
  arma::field<arma::mat> W(num_layers);
  // Bias vectors
  arma::field<arma::vec> b(num_layers);
  // Placeholder for forward propagation, initialized to X
  arma::mat A(X);

  // Read in parameter values, accumulate log-prior, and forward-prop
  unsigned int ii = 0;
  for (unsigned int l = 0; l < num_layers; l++) {
    W[l] = arma::mat(V[l+1],V[l]);
    b[l] = arma::vec(V[l+1]);
    for (unsigned int i = 0; i < W[l].n_elem; i++) {
      W[l][i] = theta[ii++];
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      b[l][i] = theta[ii++];
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
double logprob
  (
    const arma::vec& theta,
    const arma::mat& X,
    const arma::mat& B,
    const Rcpp::List& param
  )
{
  arma::vec V = param["V"]; // number of hidden neurons
  const unsigned int num_layers = V.n_elem - 1;
  string activation = param["activation"];
  Rcpp::List lambda_ = param["lambda"];

  double logprior = 0.0;
  double loglik = 0.0;

  arma::field<arma::mat> W(num_layers);
  arma::field<arma::vec> b(num_layers);
  arma::mat A(X);
  // Scale parameters for weights and bias
  //
  // W^(l)_ij ~ N(0, sigma[l]*lambda[l](j))
  // b^(l)_i ~ N(0, sigma[l]*lambda[l](0))
  arma::vec sigma = param["sigma"]; // Global layerwise scale
  arma::field<arma::vec> lambda(num_layers); // Local unitwise scale

  // Read in parameter values and accumulate log-prior
  unsigned int ii = 0;
  for (unsigned int l = 0; l < num_layers; l++) {
    lambda[l] = Rcpp::as<arma::vec>(lambda_[l]);
    lambda[l] *= sigma[l];
    W[l] = arma::mat(V[l+1],V[l]);
    b[l] = arma::vec(V[l+1]);
    for (unsigned int j = 0; j < W[l].n_cols; j++) {
      for (unsigned int i = 0; i < W[l].n_rows; i++) {
        W[l](i,j) = theta[ii++];
        logprior += R::dnorm(W[l](i,j), 0.0, lambda[l](j+1), true);
      }
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      b[l][i] = theta[ii++];
      logprior += R::dnorm(b[l][i], 0.0, lambda[l](0), true);
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
arma::vec glogprob
  (
    const arma::vec& theta,
    const arma::mat& X,
    const arma::mat& B,
    const Rcpp::List& param
  )
{
  arma::vec V = param["V"];
  const unsigned int num_layers = V.n_elem - 1;
  string activation = param["activation"];
  Rcpp::List lambda_ = param["lambda"];

  arma::field<arma::mat> W(num_layers);
  arma::field<arma::vec> b(num_layers);
  arma::field<arma::mat> A(num_layers);
  A[0] = X;
  arma::mat Z;

  arma::vec sigma = param["sigma"];
  arma::field<arma::vec> lambda(num_layers);

  // Containers for gradients
  arma::field<arma::mat> grad_W(num_layers);
  arma::field<arma::vec> grad_b(num_layers);

  unsigned int ii = 0;
  for (unsigned int l = 0; l < num_layers; l++) {
    lambda[l] = Rcpp::as<arma::vec>(lambda_[l]);
    lambda[l] *= sigma[l];
    W[l] = arma::mat(V[l+1],V[l]);
    b[l] = arma::vec(V[l+1]);
    grad_W[l] = arma::mat(V[l+1],V[l]);
    grad_b[l] = arma::vec(V[l+1]);
    for (unsigned int j = 0; j < W[l].n_cols; j++) {
      for (unsigned int i = 0; i < W[l].n_rows; i++) {
        W[l](i,j) = theta[ii++];
        grad_W[l](i,j) = -W[l](i,j) / pow(lambda[l][j+1],2);
      }
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      b[l][i] = theta[ii++];
      grad_b[l][i] = -b[l][i] / pow(lambda[l](0),2);
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
    maxv[i] = Z(maxi, i);
  }
  arma::mat enn = exp(Z.each_row() - maxv);
  arma::mat snn = B % enn;
  enn.each_row() /= sum(enn, 0);
  snn.each_row() /= sum(snn, 0);

  // Placeholder for back propagation
  arma::field<arma::mat> delta(num_layers);
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
      grad[ii++] = grad_W[l][i];
    }
    for (unsigned int i = 0; i < b[l].n_elem; i++) {
      grad[ii++] = grad_b[l][i];
    }
  }
  return grad;
}


#endif
