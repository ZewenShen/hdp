# hdp

[![Build Status](https://travis-ci.com/ZewenShen/hdp.svg?branch=master)](https://travis-ci.com/ZewenShen/hdp)

A numerical library for High-Dimensional option Pricing problems.

## Included Algorithms

#### deep Galerkin method (~/src/blackscholes/dgm)
N-d Black Scholes equation solver; 1-d American option PDE solver

#### Fourier transform methods (~/src/blackscholes/fft)
Carr & Madan algorithm; N-d Conv method

#### Monte Carlo methods (~/src/blackscholes/mc)
N-d antithetic variates; N-d control variates; N-d Sobol sequence; 1-d importance sampling; N-d Least square Monte Carlo

#### PDE (~/src/blackscholes/pde)
1-d parabolic PDE solver with Dirichlet boundary condition; 1-d American option PDE solver (PSOR / penalty method); 2-d parabolic PDE solver with Dirichlet boundary condition on a rectangle domain (untested)

#### Micellaneous (~/src/utils, ~/src/blackscholes/utils)
The analytical solution to 1-d European option and N-d geometric average payoff European option; Helper functions for conducting numerical experiments

## Reference
Z. Shen, [Numerical Methods for High-Dimensional Option Pricing Problems](https://zewenshen.github.io/files/HDP_ZewenShen.pdf)