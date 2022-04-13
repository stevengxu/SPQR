# SPQR：Semi-Parametric Quantile Regression
The `SPQR` R package implements the semi-parametric quantile regression (SPQR) method in Xu and Reich (2021) [[1]](#1). It allows flexible modeling of the conditional
disrtibution function and quantile function. The package provides three estimationg procedures: Maximum likelihood estimation (MLE) and Maximum a posteriori (MAP)
which are point estimates but computationally lighter, and Markov chain Monte Carlo (MCMC) which is fully Bayesian but computationally heavier. The MLE and MAP estimates
are obtained using the ADAM routine in `torch`, whereas the MCMC estimate is obatined using STAN-like Hamiltonian Monte Carlo (HMC) and no-U-turn sampler (NUTS).

## Installation
`devtools::install_github("stevengxu/SPQR")`

## References

<a id="1">[1]</a> 
Xu, S.G. and Reich, B.J., 2021. Bayesian nonparametric quantile process regression and estimation of marginal quantile effects. Biometrics.