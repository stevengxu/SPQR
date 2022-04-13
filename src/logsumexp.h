#ifndef LOGSUMEXP_HPP
#define LOGSUMEXP_HPP

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


/************************************
 * Helper functions for calculating
 * log of sum of exponentials
 ***********************************/

// logSumExp of a vector
double logSumExp(const arma::vec& x)
{
  unsigned int maxi = x.index_max();
  double maxv = x(maxi);
  if (!(maxv > -arma::datum::inf)) {
    return -arma::datum::inf;
  }
  double cumsum = 0.0;
  for (unsigned int i = 0; i < x.n_elem; i++) {
    if ((i != maxi) & (x(i) > -arma::datum::inf)) {
      cumsum += exp(x(i) - maxv);
    }
  }
  return maxv + log1p(cumsum);
}
// logSumExp of two numbers
double logSumExp(const double& x, const double& y)
{
  arma::vec v = {x, y};
  return logSumExp(v);
}
// logSumExp of each row of a matrix
arma::vec rowLogSumExps(const arma::mat& X)
{
  const unsigned int nrow = X.n_rows;
  arma::vec res(nrow, arma::fill::zeros);
  for (unsigned int i = 0; i < nrow; i++) {
    res(i) = logSumExp(X.row(i).t());
  }
  return res;
}
// logSumExp of each col of a matrix
arma::vec colLogSumExps(const arma::mat& X)
{
  const unsigned int ncol = X.n_cols;
  arma::vec res(ncol, arma::fill::zeros);
  for (unsigned int i = 0; i < ncol; i++) {
    res(i) = logSumExp(X.col(i));
  }
  return res;
}

#endif
