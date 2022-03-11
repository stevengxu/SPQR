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
        param(param) {}
    
    double H(const ps_point& z, const arma::vec& Minv) {
      return logprob(z.q, X, B, param) - 0.5 * dot(square(z.p), Minv);
    }
    
    double H(const ps_point& z, const arma::mat& Minv) {
      return logprob(z.q, X, B, param) - 
        0.5 * arma::as_scalar(z.p.t() * Minv * z.p);
    }
    
    void evolve(ps_point& z, const double& epsilon, const arma::vec& Minv){
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
      z.q += epsilon * (Minv % z.p);
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    }
    
    void evolve(ps_point& z, const double& epsilon, const arma::mat& Minv){
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
      z.q += epsilon * (Minv * z.p);
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    }
    
};

template <typename T>
double init_stepsize(
  const arma::vec& theta0,
  const T& Minv,
  const T& Misqrt,
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
    
    if (epsilon > 1e7)
      throw std::runtime_error(
          "Posterior is improper. "
          "Please check your model.");
    if (epsilon == 0)
      throw std::runtime_error(
          "No acceptably small step size could "
          "be found. Perhaps the posterior is "
          "not continuous?");
  }
  
  return epsilon;
}

// [[Rcpp::export]]
double init_stepsize_diag(
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
double init_stepsize_dense(
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