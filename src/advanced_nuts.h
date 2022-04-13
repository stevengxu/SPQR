#ifndef ADVANCED_NUTS_H
#define ADVANCED_NUTS_H

#include "logsumexp.h"
#include "posterior.h"
#include "ps_point.h"

bool compute_criterion
  (
    const arma::vec& p_sharp_minus,
    const arma::vec& p_sharp_plus,
    const arma::vec& rho
  )
{
  return arma::dot(p_sharp_plus, rho) > 0 && arma::dot(p_sharp_minus, rho) > 0;
}

class advanced_nuts
{

  public:

    double delta_max;
    int max_tree_depth;
    int depth;
    int n_leapfrog;
    bool divergent;
    ps_point z;
    arma::mat X;
    arma::mat B;
    Rcpp::List param;

    advanced_nuts
      (
        const arma::vec& q0,
        const arma::mat& X,
        const arma::mat& B,
        const Rcpp::List& param
      ) : delta_max(1000),
          max_tree_depth(5),
          depth(0),
          n_leapfrog(0),
          divergent(false),
          z(q0),
          X(X),
          B(B),
          param(param) { }

    void set_max_depth(int d)
    {
      if (d > 0)
        max_tree_depth = d;
    }

    void set_max_delta(double d) { delta_max = d; }

    int get_depth() { return depth; }

    bool is_divergent() { return divergent; }

    ps_point get_state() { return z; }

    double hamiltonian
      (
        const ps_point& z,
        const arma::vec& Minv
      )
    {
      return logprob(z.q, X, B, param) - 0.5 * dot(square(z.p), Minv);
    }

    double hamiltonian
      (
        const ps_point& z,
        const arma::mat& Minv
      )
    {
      return logprob(z.q, X, B, param) -
        0.5 * arma::as_scalar(z.p.t() * Minv * z.p);
    }

    void evolve
      (
        ps_point& z,
        const double& epsilon,
        const arma::vec& Minv
      )
    {
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
      z.q += epsilon * (Minv % z.p);
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    }

    void evolve
      (
        ps_point& z,
        const double& epsilon,
        const arma::mat& Minv
      )
    {
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
      z.q += epsilon * (Minv * z.p);
      z.p += 0.5 * epsilon * glogprob(z.q, X, B, param);
    }

    arma::vec dtau_dp
      (
        ps_point& z,
        const arma::vec& Minv
      )
    {
      return Minv % z.p;
    }

    arma::vec dtau_dp
      (
        ps_point& z,
        const arma::mat& Minv
      )
    {
      return Minv * z.p;
    }

    template <typename metric>
    Rcpp::List transition
      (
        const double& epsilon,
        const metric& Minv,
        const metric& Misqrt
      )
    {
      this->z.sample_p(Misqrt);
      ps_point z_fwd(this->z);
      ps_point z_bck(z_fwd);

      ps_point z_sample(z_fwd);
      ps_point z_propose(z_fwd);

      arma::vec p_fwd_fwd = this->z.p;
      arma::vec p_sharp_fwd_fwd = this->dtau_dp(this->z, Minv);

      arma::vec p_fwd_bck = this->z.p;
      arma::vec p_sharp_fwd_bck = p_sharp_fwd_fwd;

      arma::vec p_bck_fwd = this->z.p;
      arma::vec p_sharp_bck_fwd = p_sharp_fwd_fwd;

      arma::vec p_bck_bck = this->z.p;
      arma::vec p_sharp_bck_bck = p_sharp_fwd_fwd;

      arma::vec rho = this->z.p;

      // Log sum of state weights (offset by H0) along trajectory
      double log_sum_weight = 0;
      double H0 = this->hamiltonian(this->z, Minv);
      int n_leapfrog = 0;
      double sum_metro_prob = 0;

      // Build a trajectory until the no-u-turn
      // criterion is no longer satisfied
      while (this->depth < this->max_tree_depth) {
        arma::vec rho_fwd(rho.n_elem, arma::fill::zeros);
        arma::vec rho_bck(rho.n_elem, arma::fill::zeros);

        bool valid_subtree = false;
        double log_sum_weight_subtree = -std::numeric_limits<double>::infinity();

        if (R::runif(0,1) > 0.5) {
          this->z = z_fwd;
          rho_bck = rho;
          p_bck_fwd = p_fwd_fwd;
          p_sharp_bck_fwd = p_sharp_fwd_fwd;

          valid_subtree =
            build_tree(this->depth, z_propose, p_sharp_fwd_bck, p_sharp_fwd_fwd,
                       rho_fwd, p_fwd_bck, p_fwd_fwd, H0, 1, n_leapfrog,
                       log_sum_weight_subtree, sum_metro_prob, epsilon, Minv);
          z_fwd = this->z;
        } else {
          this->z = z_bck;
          rho_fwd = rho;
          p_fwd_bck = p_bck_bck;
          p_sharp_fwd_bck = p_sharp_bck_bck;

          valid_subtree =
            build_tree(this->depth, z_propose, p_sharp_bck_fwd, p_sharp_bck_bck,
                       rho_bck, p_bck_bck, p_fwd_fwd, H0, -1, n_leapfrog,
                       log_sum_weight_subtree, sum_metro_prob, epsilon, Minv);
          z_bck = this->z;
        }

        if (!valid_subtree)
          break;

        ++(this->depth);

        if (log_sum_weight_subtree > log_sum_weight) {
          z_sample = z_propose;
        } else {
          double accept_prob = exp(log_sum_weight_subtree - log_sum_weight);
          if (R::runif(0,1) < accept_prob)
            z_sample = z_propose;
        }

        log_sum_weight = logSumExp(log_sum_weight, log_sum_weight_subtree);

        // Break when no-u-turn criterion is no longer satisfied
        rho = rho_bck + rho_fwd;

        // Demand satisfaction around merged subtrees
        bool persist_criterion =
          compute_criterion(p_sharp_bck_bck, p_sharp_fwd_fwd, rho);

        // Demand satisfaction between subtrees
        arma::vec rho_extended = rho_bck + p_fwd_bck;

        persist_criterion &=
          compute_criterion(p_sharp_bck_bck, p_sharp_fwd_bck, rho_extended);

        rho_extended = rho_fwd + p_bck_fwd;
        persist_criterion &=
          compute_criterion(p_sharp_bck_fwd, p_sharp_fwd_fwd, rho_extended);

        if (!persist_criterion)
          break;
      }

      this->n_leapfrog = n_leapfrog;

      // Compute average acceptance probabilty across entire trajectory,
      // even over subtrees that may have been rejected
      double accept_prob = sum_metro_prob / static_cast<double>(n_leapfrog);

      this->z = z_sample;
      Rcpp::List out =
        Rcpp::List::create(Rcpp::Named("theta") = this->z.q,
                           Rcpp::Named("accept.prob") = accept_prob);
      return out;
    }

    template <typename metric>
    bool build_tree
      (
        const int& depth,
        ps_point& z_propose,
        arma::vec& p_sharp_beg,
        arma::vec& p_sharp_end,
        arma::vec& rho,
        arma::vec& p_beg,
        arma::vec& p_end,
        const double H0,
        const double sign,
        int& n_leapfrog,
        double& log_sum_weight,
        double& sum_metro_prob,
        const double& epsilon,
        const metric& Minv
      )
    {
      //Base case
      if (depth == 0) {
        this->evolve(this->z, sign * epsilon, Minv);

        ++n_leapfrog;

        double h = this->hamiltonian(this->z, Minv);
        if (std::isnan(h))
          h = -std::numeric_limits<double>::infinity();

        if ((H0 - h) > this->delta_max)
          this->divergent = true;

        log_sum_weight = logSumExp(log_sum_weight, h - H0);

        if (h - H0 > 0)
          sum_metro_prob += 1;
        else
          sum_metro_prob += exp(h - H0);

        z_propose = this->z;

        p_sharp_beg = this->dtau_dp(this->z, Minv);
        p_sharp_end = p_sharp_beg;

        rho += this->z.p;
        p_beg = this->z.p;
        p_end = p_beg;

        return !this->divergent;
      }
      // General recursion

      // Build the initial subtree
      double log_sum_weight_init = -std::numeric_limits<double>::infinity();

      // Momentum and sharp momentum at end of the initial subtree
      arma::vec p_init_end(this->z.npar);
      arma::vec p_sharp_init_end(this->z.npar);

      arma::vec rho_init(rho.n_elem, arma::fill::zeros);

      bool valid_init =
        build_tree(depth - 1, z_propose, p_sharp_beg, p_sharp_init_end,
                   rho_init, p_beg, p_init_end, H0, sign, n_leapfrog,
                   log_sum_weight_init, sum_metro_prob, epsilon, Minv);
      if (!valid_init)
        return false;

      // Build the final subtree
      ps_point z_propose_final(this->z);

      double log_sum_weight_final = -std::numeric_limits<double>::infinity();

      // Momentum and sharp momentum at beginning of the final subtree
      arma::vec p_final_beg(this->z.npar);
      arma::vec p_sharp_final_beg(this->z.npar);

      arma::vec rho_final(rho.n_elem, arma::fill::zeros);

      bool valid_final =
        build_tree(depth - 1, z_propose_final, p_sharp_final_beg, p_sharp_end,
                   rho_final, p_final_beg, p_end, H0, sign, n_leapfrog,
                   log_sum_weight_final, sum_metro_prob, epsilon, Minv);
      if (!valid_final)
        return false;

      // Multinomial sample from right subtree
      double log_sum_weight_subtree =
        logSumExp(log_sum_weight_init, log_sum_weight_final);
      log_sum_weight = logSumExp(log_sum_weight, log_sum_weight_subtree);

      if (log_sum_weight_final > log_sum_weight_subtree) {
        z_propose = z_propose_final;
      } else {
        double accept_prob = exp(log_sum_weight_final - log_sum_weight_subtree);
        if (R::runif(0,1) < accept_prob)
          z_propose = z_propose_final;
      }

      arma::vec rho_subtree = rho_init + rho_final;
      rho += rho_subtree;

      // Demand satisfaction around merged subtrees
      bool persist_criterion =
        compute_criterion(p_sharp_beg, p_sharp_end, rho_subtree);

      // Demand satisfaction between subtrees
      rho_subtree = rho_init + p_final_beg;
      persist_criterion &=
        compute_criterion(p_sharp_beg, p_sharp_final_beg, rho_subtree);

      rho_subtree = rho_final + p_init_end;
      persist_criterion &=
        compute_criterion(p_sharp_init_end, p_sharp_end, rho_subtree);

      return persist_criterion;
    }
};


#endif
