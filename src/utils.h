// Inline functions to convert between Rcpp and Arma data structures

#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// convert arma matrix type to (column) vector type
inline arma::vec mat2vec(const arma::mat& x)
{
  return arma::conv_to<arma::vec>::from(x);
}
// convert arma matrix type to row vector type
inline arma::rowvec mat2rowvec(const arma::mat& x)
{
  return arma::conv_to<arma::rowvec>::from(x);
}
// convert arma vector type to double type
inline arma::vec num2vec(const double x)
{
  arma::vec out { arma::zeros(1) };
  out(0) = x;
  return out;
}
// convert arma vector type to Rcpp vector type
template <typename T>
inline Rcpp::NumericVector arma2rvec(const T& x)
{
  return Rcpp::NumericVector(x.begin(), x.end());
}
// convert Rcpp::NumericVector to arma::colvec
template <typename T>
inline arma::vec rvec2arma(const T& x)
{
  return arma::vec(x.begin(), x.size(), false);
}
// convert arma matrix type to Rcpp matrix type
inline Rcpp::NumericMatrix arma2rmat(const arma::mat& x)
{
  return Rcpp::NumericMatrix(x.n_rows, x.n_cols, x.begin());
}
// convert Rcpp matrix to arma matrix
inline arma::mat rmat2arma(Rcpp::NumericMatrix& x)
{
  return arma::mat(x.begin(), x.nrow(), x.ncol(), false);
}


#endif
