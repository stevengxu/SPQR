#ifndef INIT_STEPSIZE_H
#define INIT_STEPSIZE_H


#include "posterior.h"
#include "ps_point.h"

class Hamiltonian {
public:

  arma::mat X;
  arma::mat B;
  Rcpp::List param;

  Hamiltonian(const arma::mat& X,
              const arma::mat& B,
              const Rcpp::List& param)
    : X(X),
      B(B),
      param(param) { }

  double H(const ps_point& z, const arma::vec& Minv)
  {
    return logprob(z.q, X, B, param) - 0.5 * dot(square(z.p), Minv);
  }

  double H(const ps_point& z, const arma::mat& Minv)
  {
    return logprob(z.q, X, B, param) -
      0.5 * arma::as_scalar(z.p.t() * Minv * z.p);
  }

  void evolve(ps_point& z, const double& epsilon, const arma::vec& Minv)
  {
    z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    z.q += epsilon * (Minv % z.p);
    z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
  }

  void evolve(ps_point& z, const double& epsilon, const arma::mat& Minv)
  {
    z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    z.q += epsilon * (Minv * z.p);
    z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
  }

};

template <typename metric>
double init_stepsize
  (
    const arma::vec& theta0,
    const metric& Minv,
    const metric& Misqrt,
    const arma::mat& X,
    const arma::mat& B,
    const Rcpp::List& param
  )
{
  double epsilon = 0.01;

  ps_point z(theta0);
  ps_point z_init(z);
  z.sample_p(Misqrt);

  Hamiltonian hamiltonian(X, B, param);

  double H0 = hamiltonian.H(z, Minv);

  hamiltonian.evolve(z, epsilon, Minv);

  double h = hamiltonian.H(z, Minv);
  if (std::isnan(h))
    h = std::numeric_limits<double>::infinity();

  double delta_H = H0 - h;

  int direction = delta_H > log(0.8) ? 1 : -1;

  while (1) {
    z = z_init;
    z.sample_p(Misqrt);

    double H0 = hamiltonian.H(z, Minv);

    hamiltonian.evolve(z, epsilon, Minv);

    double h = hamiltonian.H(z, Minv);
    if (std::isnan(h))
      h = std::numeric_limits<double>::infinity();

    double delta_H = H0 - h;

    if ((direction == 1) && !(delta_H > log(0.8)))
      break;
    else if ((direction == -1) && !(delta_H < std::log(0.8)))
      break;
    else
      epsilon = direction == 1 ? 2.0 * epsilon : 0.5 * epsilon;

    if (epsilon > 1) {
      epsilon = 1;
      break;
    }

    if (epsilon < 1e-8) {
      epsilon = 1e-8;
      break;
    }
  }

  return epsilon;
}

#endif
