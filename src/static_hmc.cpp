#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "utils.h"
#include "logsumexp.h"
#include "posterior.h"
#include "ps_point.h"

class static_hmc {
  public:
    
    double T;
    int L;
    bool divergent;
    ps_point z;
    arma::mat X;
    arma::mat B;
    Rcpp::List param;
    
    static_hmc(const arma::vec& q0, 
               const arma::mat& X, 
               const arma::mat& B, 
               const Rcpp::List param)
      : T(1),
        divergent(false),
        z(q0),
        X(X),
        B(B),
        param(param) {}
    
    void set_T(const double& t) {
      if (t > 0) 
        T = t;
    }
    
    void update_L(const double& epsilon) {
      L = static_cast<int>(T / epsilon);
      L = L < 1 ? 1 : L;
    }
    
    int get_L() { return this->L; }
    
    bool is_divergent() { return this->divergent; }
    
    double hamiltonian(
        const ps_point& z,
        const arma::vec& Minv
        )
    {
      return logprob(z.q, X, B, param) - 0.5 * dot(square(z.p), Minv);
    }
    
    double hamiltonian(
        const ps_point& z,
        const arma::mat& Minv
        )
    {
      return logprob(z.q, X, B, param) - 
        0.5 * arma::as_scalar(z.p.t() * Minv * z.p);
    }
    
    void evolve(
        ps_point& z,
        const double& epsilon,
        const arma::vec& Minv
        )
    {
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
      z.q += epsilon * (Minv % z.p);
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    }
    
    void evolve(
        ps_point& z,
        const double& epsilon,
        const arma::mat& Minv
        )
    {
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
      z.q += epsilon * (Minv * z.p);
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    }
    
    template <typename T>
    Rcpp::List transition(
        const double& epsilon,
        const T& Minv,
        const T& Misqrt
        )
    {
      this->z.sample_p(Misqrt);
      
      ps_point z_init(this->z);
      
      double H0 = this->hamiltonian(this->z, Minv);
      
      for (int i = 0; i < this->L; ++i)
        this->evolve(this->z, epsilon, Minv);
      
      double h = this->hamiltonian(this->z, Minv);
      if (std::isnan(h)) {
        this->divergent = true;
        h = std::numeric_limits<double>::infinity();
      }
        
      double acceptProb = std::min(1.0, exp(h - H0));
      
      if (R::runif(0,1) > acceptProb)
        this->z = z_init;
      
      Rcpp::List out = 
        Rcpp::List::create(Rcpp::Named("theta") = this->z.q,
                           Rcpp::Named("accept_prob") = acceptProb);
      return out;
    }
};

// [[Rcpp::export]]
Rcpp::List static_hmc_diag(
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
  
  info["num_steps__"] = sampler.get_L();
  if (sampler.is_divergent())
    info["divergent__"] = 1;
  
  return res;
}

// [[Rcpp::export]]
Rcpp::List static_hmc_dense(
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
  
  info["num_steps__"] = sampler.get_L();
  if (sampler.is_divergent())
    info["divergent__"] = 1;
  
  return res;
}
