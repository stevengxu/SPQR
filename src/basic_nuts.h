#ifndef BASIC_NUTS_H
#define BASIC_NUTS_H

#include "logsumexp.h"
#include "posterior.h"
#include "ps_point.h"

bool compute_criterion(
    const ps_point& start,
    const ps_point& finish,
    const arma::vec& rho,
    const arma::vec& Minv
    )
{
  return arma::dot(Minv % finish.p, rho - finish.p) > 0 &&
    arma::dot(Minv % finish.p, rho - start.p) > 0;
}

bool compute_criterion(
    const ps_point& start,
    const ps_point& finish,
    const arma::vec& rho,
    const arma::mat& Minv
    )
{
  return arma::as_scalar(finish.p.t() * Minv * (rho - finish.p)) > 0 &&
    arma::as_scalar(start.p.t() * Minv * (rho - start.p)) > 0;
}

struct nuts_util {
  double logu;
  double H0;
  int sign;

  int n_tree;
  double sum_prob;
  bool criterion;

  nuts_util() : criterion(false) {}
};

class basic_nuts {
  public:

    double delta_max;
    int max_tree_depth;
    int depth;
    bool divergent;
    ps_point z;
    arma::mat X;
    arma::mat B;
    Rcpp::List param;

    basic_nuts(const arma::vec& q0,
               const arma::mat& X,
               const arma::mat& B,
               const Rcpp::List& param)
      : delta_max(1000),
        max_tree_depth(5),
        depth(0),
        divergent(false),
        z(q0),
        X(X),
        B(B),
        param(param){}

    void set_max_depth(int d) {
      if (d > 0)
        max_tree_depth = d;
    }

    void set_max_delta(double d) { delta_max = d; }

    bool is_divergent() { return divergent; }

    ps_point get_state() { return z; }

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

      nuts_util util;

      ps_point z_plus(this->z);
      ps_point z_minus(z_plus);

      ps_point z_sample(z_plus);
      ps_point z_propose(z_plus);

      int npar = this->z.npar;

      arma::vec rho_init = this->z.p;
      arma::vec rho_plus(npar, arma::fill::zeros);
      arma::vec rho_minus(npar, arma::fill::zeros);

      util.H0 = this->hamiltonian(this->z, Minv);

      // Sample the slice variable
      util.logu = log(R::runif(0,1));

      // Build a balanced binary tree until the NUTS criterion fails
      util.criterion = true;
      int n_valid = 0;

      util.n_tree = 0;
      util.sum_prob = 0;

      while (util.criterion && (this->depth <= this->max_tree_depth)) {

        ps_point* z_ = 0;
        arma::vec* rho = 0;

        if (R::runif(0,1) > 0.5) {
          z_ = &z_plus;
          rho = &rho_plus;
          util.sign = 1;
        } else {
          z_ = &z_minus;
          rho = &rho_minus;
          util.sign = -1;
        }

        // And build a new subtree in that direction
        this->z = *z_;

        int n_valid_subtree =
          build_tree(this->depth, *rho, 0, z_propose, util, epsilon, Minv);
        ++(this->depth);

        // Metropolis-Hastings sample the fresh subtree
        if (!util.criterion)
          break;


        double subtree_prob = 0;

        if (n_valid) {
          subtree_prob = static_cast<double>(n_valid_subtree)
          / static_cast<double>(n_valid);
        } else {
          subtree_prob = n_valid_subtree ? 1 : 0;
        }

        if (R::runif(0,1) < subtree_prob)
          z_sample = z_propose;

        n_valid += n_valid_subtree;

        // Check validity of completed tree
        this->z = z_plus;
        arma::vec delta_rho = rho_minus + rho_init + rho_plus;

        util.criterion = compute_criterion(z_minus, this->z, delta_rho, Minv);
      }
      double accept_prob = util.sum_prob / static_cast<double>(util.n_tree);
      this->z = z_sample;
      Rcpp::List out =
        Rcpp::List::create(Rcpp::Named("theta") = this->z.q,
                           Rcpp::Named("accept.prob") = accept_prob);
      return out;

    }

    template <typename T>
    int build_tree(
        const int& depth,
        arma::vec& rho,
        ps_point* z_init_parent,
        ps_point& z_propose,
        nuts_util& util,
        const double& epsilon,
        const T& Minv
        )
    {
      if (depth == 0) {
        // Base case: Take a single leapfrog step in the direction `v`
        this->evolve(this->z, util.sign*epsilon, Minv);

        rho += this->z.p;

        if (z_init_parent)
          *z_init_parent = this->z;
        z_propose = this->z;

        double h = this->hamiltonian(this->z, Minv);
        if (std::isnan(h))
          h = std::numeric_limits<double>::infinity();

        // Is the new point in the slice?
        util.criterion = (util.logu + (util.H0 - h)) < this->delta_max;
        if (!util.criterion)
          this->divergent = true;

        // Acceptance ratio
        util.sum_prob += std::min(1.0, exp(h - util.H0));
        util.n_tree += 1;

        return (util.logu + (util.H0 - h) < 0);

      } else {

        arma::vec left_subtree_rho(rho.n_elem, arma::fill::zeros);
        ps_point z_init(this->z);

        unsigned int n1 = build_tree(depth - 1, left_subtree_rho, &z_init,
                                     z_propose, util, epsilon, Minv);

        if (z_init_parent)
          *z_init_parent = z_init;

        if (!util.criterion)
          return 0;

        arma::vec right_subtree_rho(rho.n_elem, arma::fill::zeros);
        ps_point z_propose_right(z_init);

        unsigned int n2 = build_tree(depth - 1, right_subtree_rho, 0,
                                     z_propose_right, util, epsilon, Minv);

        double accept_prob
          = static_cast<double>(n2) / static_cast<double>(n1 + n2);

        if (util.criterion && (R::runif(0,1) < accept_prob))
          z_propose = z_propose_right;

        arma::vec& subtree_rho = left_subtree_rho;
        subtree_rho += right_subtree_rho;

        rho += subtree_rho;

        util.criterion &= compute_criterion(z_init, this->z, subtree_rho, Minv);

        return n1 + n2;
      }
    }
};


Rcpp::List basic_nuts_diag(
    const arma::vec& q0,
    const arma::mat& X,
    const arma::mat& B,
    const Rcpp::List& param,
    const double& epsilon,
    const arma::vec& Minv,
    const arma::vec& Misqrt,
    const int max_tree_depth,
    Rcpp::List& info
)
{
  basic_nuts sampler(q0, X, B, param);
  sampler.set_max_depth(max_tree_depth);

  Rcpp::List res = sampler.transition(epsilon, Minv, Misqrt);

  if (sampler.is_divergent())
    info["divergent"] = 1;

  return res;
}


Rcpp::List basic_nuts_dense(
    const arma::vec& q0,
    const arma::mat& X,
    const arma::mat& B,
    const Rcpp::List& param,
    const double& epsilon,
    const arma::mat& Minv,
    const arma::mat& Misqrt,
    const int max_tree_depth,
    Rcpp::List& info
)
{
  basic_nuts sampler(q0, X, B, param);
  sampler.set_max_depth(max_tree_depth);

  Rcpp::List res = sampler.transition(epsilon, Minv, Misqrt);

  if (sampler.is_divergent())
    info["divergent"] = 1;

  return res;
}

#endif
