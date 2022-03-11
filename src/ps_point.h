#ifndef PS_POINT_H
#define PS_POINT_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

class ps_point {
public:
  arma::vec q;
  arma::vec p;
  unsigned int npar;
  
  ps_point(const arma::vec& q) {
    this->q = q;
    p.zeros(q.n_elem);
    npar = q.n_elem;
  }
  
  ps_point(const ps_point& point) {
    q = point.q;
    p = point.p;
    npar = q.n_elem;
  }
  
  void operator = (const ps_point& point) {
    q = point.q;
    p = point.p;
  }
  
  void sample_p(const arma::vec& Misqrt) {
    for (unsigned int i = 0; i < p.n_elem; i++) {
      p[i] = R::rnorm(0,1);
    }
    p /= Misqrt;
  }
  
  void sample_p(const arma::mat& Misqrt) {
    for (unsigned int i = 0; i < p.n_elem; i++) {
      p[i] = R::rnorm(0,1);
    }
    p = solve(arma::trimatu(Misqrt),p);
  }
};

#endif