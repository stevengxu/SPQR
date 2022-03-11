#ifndef SOFTMAX_H
#define ACTIVATION_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// softmax transformation of a vector
arma::vec softmax(const arma::vec& x) {
  arma::vec s(x.n_elem);
  unsigned int maxi = x.index_max();
  double maxv = x(maxi);
  
  double cumsum = 0.0;
  for (unsigned int i = 0; i < x.n_elem; i++) {
    s(i) = exp(x(i) - maxv);
    cumsum += s(i);
  }
  
  s /= cumsum;
  return s;
}

// softmax transformation of each column of a matrix
arma::mat colSoftmax(const arma::mat& X) {
  unsigned int nrow = X.n_rows;
  unsigned int ncol = X.n_cols;
  arma::mat S(nrow, ncol);
  for (unsigned int i = 0; i < ncol; i++) {
    S.col(i) = softmax(X.col(i));
  }
  return S;
}

// relu activation of each element of a matrix
arma::mat relu(const arma::mat& X) {
  arma::mat Y(X);
  arma::uvec idx = arma::find(Y < 0.0);
  if (idx.n_elem > 0) {
    Y(idx).fill(0.0);
  }
  return Y;
}

// gradient of relu
arma::mat relu_grad(const arma::mat& X) {
  arma::mat Y(X);
  arma::uvec idx = arma::find(Y > 0.0);
  if (idx.n_elem > 0) {
    Y(idx).fill(1.0);
  }
  return Y;
}

#endif