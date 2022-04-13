#include "init_stepsize.h"
#include "advanced_nuts.h"
#include "static_hmc.h"

// [[Rcpp::export]]
double rcpp_init_stepsize_diag
  (
      const arma::vec& theta0,
      const arma::vec& Minv,
      const arma::vec& Misqrt,
      const arma::mat& X,
      const arma::mat& B,
      const Rcpp::List& param
  )
{
  return init_stepsize(theta0, Minv, Misqrt, X, B, param);
}

// [[Rcpp::export]]
double rcpp_init_stepsize_dense
  (
      const arma::vec& theta0,
      const arma::mat& Minv,
      const arma::mat& Misqrt,
      const arma::mat& X,
      const arma::mat& B,
      const Rcpp::List& param
  )
{
  return init_stepsize(theta0, Minv, Misqrt, X, B, param);
}


// [[Rcpp::export]]
Rcpp::List rcpp_nuts_diag
  (
      const arma::vec& q0,
      const arma::mat& X,
      const arma::mat& B,
      const Rcpp::List& param,
      const double& epsilon,
      const arma::vec& Minv,
      const arma::vec& Misqrt,
      const int& max_tree_depth,
      Rcpp::List& info
  )
{
  advanced_nuts sampler(q0, X, B, param);
  sampler.set_max_depth(max_tree_depth);

  Rcpp::List res = sampler.transition(epsilon, Minv, Misqrt);

  info["treedepth"] = sampler.get_depth();
  if (sampler.is_divergent())
    info["divergent"] = 1;

  return res;
}

// [[Rcpp::export]]
Rcpp::List rcpp_nuts_dense
  (
      const arma::vec& q0,
      const arma::mat& X,
      const arma::mat& B,
      const Rcpp::List& param,
      const double& epsilon,
      const arma::mat& Minv,
      const arma::mat& Misqrt,
      const int& max_tree_depth,
      Rcpp::List& info
  )
{
  advanced_nuts sampler(q0, X, B, param);
  sampler.set_max_depth(max_tree_depth);

  Rcpp::List res = sampler.transition(epsilon, Minv, Misqrt);

  info["treedepth"] = sampler.get_depth();
  if (sampler.is_divergent())
    info["divergent"] = 1;

  return res;
}

// [[Rcpp::export]]
Rcpp::List rcpp_hmc_diag
  (
      const arma::vec& q0,
      const arma::mat& X,
      const arma::mat& B,
      const Rcpp::List& param,
      const double& epsilon,
      const arma::vec& Minv,
      const arma::vec& Misqrt,
      const double& int_time,
      Rcpp::List& info
  )
{
  static_hmc sampler(q0, X, B, param);
  sampler.set_T(int_time);
  sampler.update_L(epsilon);

  Rcpp::List res = sampler.transition(epsilon, Minv, Misqrt);

  info["num.steps"] = sampler.get_L();
  if (sampler.is_divergent())
    info["divergent"] = 1;

  return res;
}

// [[Rcpp::export]]
Rcpp::List rcpp_hmc_dense
  (
      const arma::vec& q0,
      const arma::mat& X,
      const arma::mat& B,
      const Rcpp::List& param,
      const double& epsilon,
      const arma::mat& Minv,
      const arma::mat& Misqrt,
      const double& int_time,
      Rcpp::List& info
  )
{
  static_hmc sampler(q0, X, B, param);
  sampler.set_T(int_time);
  sampler.update_L(epsilon);

  Rcpp::List res = sampler.transition(epsilon, Minv, Misqrt);

  info["num.steps"] = sampler.get_L();
  if (sampler.is_divergent())
    info["divergent"] = 1;

  return res;
}

