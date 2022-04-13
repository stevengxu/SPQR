#ifndef PS_POINT_HPP
#define PS_POINT_HPP

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


/************************************
 * Phase point class
 * Stores info of current transition
 ***********************************/
class ps_point
{

  public:

    arma::vec q;
    arma::vec p;
    unsigned int npar;

    // *******************************
    // Constructors
    // *******************************

    ps_point(const arma::vec& q)
    {
      this->q = q;
      p.zeros(q.n_elem);
      npar = q.n_elem;
    }
    ps_point(const ps_point& point)
    {
      q = point.q;
      p = point.p;
      npar = q.n_elem;
    }

    // compares two phase points
    void operator = (const ps_point& point)
    {
      q = point.q;
      p = point.p;
    }

    // sample auxillary variable for dense mass matrix
    void sample_p(const arma::vec& Misqrt)
    {
      for (unsigned int i = 0; i < p.n_elem; i++) {
        p[i] = R::rnorm(0,1);
      }
      p /= Misqrt;
    }
    // sample auxillary variable for diagonal or unit mass matrix
    void sample_p(const arma::mat& Misqrt)
    {
      for (unsigned int i = 0; i < p.n_elem; i++) {
        p[i] = R::rnorm(0,1);
      }
      p = solve(arma::trimatu(Misqrt),p);
    }

};

#endif
