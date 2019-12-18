
#  7.31

##  Progress

chapter 6.1 - 6.4 of Chen's thesis. chapter 1 - 4.1 of the foundation work in this field: [the deep Galerkin method (DGM)](https://arxiv.org/abs/1708.07469), http://utstat.toronto.edu/~ali/papers/PDEandDeepLearning.pdf as a supplementary material.

###  Cholesky decomposition for correlation matrix

Corr = $\Rho = LL^T$. $dW = LN\sqrt{dt}$ where $N$ is a normal distribution RV vector. In such casem we will have $$corr(dW_i, dW_j) = E[dw_idw_j]-E[dw_i]E[dw_j] = E[dw_idw_j]-0 = \Rho_{ij}$$

  

###  Feyman-Kac can transform second order linear pde into bsde

  
#  8.7

##  Progress

Read Chen's thesis in detail. Read [Longstaff Schwartz paper](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf). Learned the basic of pytorch and keras.

###  Complete delta heding

During the delta hedging process, we need Delta at each time step for the current stock price. We can retrieve that either by running the MC again, or interpolate using results gained from previous MC.

#  8.14

##  Progress

Implemented the Longstaff-Schwartz method. Read Chen's thesis in a great detail.

###  F1 score

F1 score is a good indicator of accuracy when the proportion of false positive and true negative is skewed.

$$F_1 = 2\frac{precision\cdot recall}{precision+recall}$$

where $precision = \frac{tp}{tp+fp}$, $recall=\frac{tp}{tp+fn}$.

  

###  TODO: P&L why not centre at x=1?

  

###  Use delta hedging to verify the accuracy of Delta?

  

###  Figured out why Chen's multi-step architecture is more efficient

There are less neural networks in a single pricing model.

  

#  8.23

  

##  Progress

Implemented antithetic variates, control variates to reduce variation in MC. Learned PCA in detail. Read [Dimension Reduction for the

Black-Scholes Equation](https://www.it.uu.se/edu/course/homepage/projektTDB/vt07/Presentationer/Projekt3/Dimension_Reduction_for_the_Black-Scholes_Equation.pdf) to figure out one potential method that deals with high dimensional PDEs.

  

###  Overfitting of LSMC by choosing high order monomial basis

We need to guarantee that the basis # is much smaller than the sample size. It's also recommended to add a regularization term:

Initially we want $min_c ||Ac- y||_2^2$, now we add a regularization term $\rho||c||_2^2$. (Note that sometimes we use 1-norm for the regularization term, in which case we have to solve it using iterative methods.) So it becomes

$$min_c (||Ac- y||_2^2+\rho||c||_2^2) $$

$$=min_c(||\Big[A\ \ I\Big]^Tc - \Big[y \ \ 0\Big]^T||)$$

which again can be solved by QR or SVD.

The analytical form of original sol is $(A^TA)c = A^Ty$.

  

###  It's more accurate to simulate stock price by a kinda analytical sol instead of the crude Euler method

$$S_t = S_0e^{(r-\sigma^2/2)t + \sigma\sqrt {t} z}, where\ Z\sim N(0,1).$$

European option pricing can be accelerated greatly by this, since we can directly get the price at time T.

#####  TODO: Generalize it into multi-asset version. (9.3 Done)

  

###  Variance decomposition of antithetic variates

In P208 Glasserman, we decompose $f(z)$ into an even function $f_0(z) = \frac{f(z)+f(-z)}{2}$ and an odd function $f_1(z) = \frac{f(z)-f(-z)}{2}$. We show that $f_0$ and $f_1$ are uncorrelated:

$$E[f_0(Z)f_1(Z)] = \frac{1}{4}E[f^2(Z)-f^2(-Z)]$$

Let the pdf of Z be $p(z)$,

$$E[f^2(Z)] = \int_{-\infty}^\infty f^2(x)p(x)dx = -\int_{\infty}^{-\infty} f^2(-x)p(-x)dx = \int_{-\infty}^\infty f^2(-x)p(-x)dx=E[f^2(-Z)] $$

So

$$E[f_0(Z)f_1(Z)] = 0.$$

It follows that

$$Var[f(Z)] = Var[f_0(Z)] + Var[f_1(Z)].$$

The first term on the right is the variance of an estimate of $E[f(Z)]$ based on an antithetic pair $(Z, -Z)$. So if $f$ is very odd ($||f-f_1||$ is small), the antithetic variates would work well. Vice versa.

  
  

###  PCA covariance matrix

In the usual problem, the covariance matrix needs to be approximated. But in option pricing, we assume that a correlation matrix is given. In this case, we can recover the correlation matrix to a covariance matrix by only approximating variance of stock price (TODO: Can be analytically expressed?). It should give a more accurate price.

  
  

#  9.3

  

##  Progress

Implemented the analytical solution to GBM. Implemented importance sampling for European options. Implemented 1D parabolic PDE solver. Halfway through the implementation of 2D parabolic solver. PDE change of variable. Martingale.

  

### Tensor product 
$(A\otimes B)(C\otimes D)=AC\otimes BD$

$A\otimes B$ can be interpreted as the composition of two finite difference matrices where A deals with X axis, B deals with Y axis. For example, when we want to approximate $u_{xx}$, we have $T_{2,S_1}\otimes I_{S_2}$

It can also be used in numerical linalg. Suppose we are going to solve $(A\otimes I) x = b$. We know that $A\otimes I$ will be a large matrix that's hard to be LU factorized. So we factorize A to be $LU$. Then lhs becomes $(LU\otimes I) x = (L\otimes I)(U\otimes I)x$. Since $(L\otimes I)$ and $(U\otimes I)$ are still lower trian and upper trian, this is easy to solve.

### Change of variable in PDE
Consider the BS equation $\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial ^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0, \quad 0\leq S\leq \infty, t\leq T$.

Let $S=e^x$. Then $\frac{\partial S}{\partial x} = e^x = S$
$\frac{\partial V}{\partial S} = \frac{\partial V}{\partial x}\frac{\partial x}{\partial S} =\frac{1}{S}\frac{\partial V}
{\partial x}$

$\frac{\partial^2 V}{\partial S^2} = \frac{\partial }{\partial S}(\frac{\partial V}{\partial S}) = \frac{\partial }{\partial S}(\frac{1}{S}\frac{\partial V}
{\partial x})=-\frac{1}{S^2}\frac{\partial V}{\partial x}+\frac{1}{S}\frac{\partial}{\partial S}(\frac{\partial V}{\partial x})=-\frac{1}{S^2}\frac{\partial V}{\partial x}+\frac{1}{S}\frac{\partial x}{\partial S}\frac{\partial}{\partial x}(\frac{\partial V}{\partial x})=-\frac{1}{S^2}\frac{\partial V}{\partial x}+\frac{1}{S^2}\frac{\partial^2 V}{\partial x^2}$

# 9.10

## Progress

Further implementation on 2D parabolic pde solver (haven't debugged). Read chapter 3 (probability & information theory), chapter 5 (machine learning basics), half of chapter 6 (deep feedforward networks). Studied concepts in information theory in depth. Read again chapter 1 - 4.1 of the foundation work in this field: [the deep Galerkin method (DGM)](https://arxiv.org/abs/1708.07469), http://utstat.toronto.edu/~ali/papers/PDEandDeepLearning.pdf as a supplementary material.

# 9.17

## Progress

Learned the use of tensorflow. Implemented the deep galerkin method for single-asset BS equation. Came up with several ideas about how to improve DGM. Method of characteristic. Read "Numerical methods for conservation laws", learned some mathematical foundation of the conservation laws and the numerical difficulties. Learned some numerical methods for Burgers' equation. 

### DGM possible improvements
Add penalty terms (1. make delta monotonic. 2. avoid negative or > stock price option) to th loss term. Do more samplings in BC and IC at the beginning.

### Advantage of DGM
Able to get Greek on the whole domain. Able to know the loss when computing solutions.

# 9.24

## Progress

Made some helper functions for deep learning experiments. Attempts: 1. add another penalty term that requires the first derivative to be always non-positive or non-negative; 2. consider boundary conditions in the loss function. Learned heat equation's properties. Proved the error bound for the deep Galerkin method given the loss on the whole domain.

# 10.1

## Progress

Explored maximum principle in depth. Computed the error bound for the n dim inhomogeneous Black-Scholes equation. Learned basic discrete fourier transformation.

# 10.8

## Progress

Learned regular/discrete Fourier transform and FFT. Learned the characteristic function of probability densities. Read Option valuation using the fast Fourier transform (Carr & Madan) using Efficient Options Pricing Using the Fast Fourier Transform as supplementary material. Implemented Carr & Madan method for 1D European option.

### Why log transform of S when Fourier time stepping is used
Such that we can get a constant variable PDE, which will be the target of Fourier time stepping method. And theoretically, the target function should be defined on R to make it Fourier transformable.

### Fourier transform & Fourier inverse transform are basically the same thing. If we did the inverse trans first, the resulting function will also be in the frequency domain

### TODO: Why 0 point must be picked in DFT?

### Idea of Carr-Madan method
If the characteristic function of the density function is known, we can give the fourier transformed option price in an analytical form. Then we can do FFT to transform it back to real price.

### The complex part of approximated solution
Mathematically speaking, there would be no complex part, since Fourier + Fourier inverse give the original option price, which is real. But truncation error would possibly lead to complex number. In this case, we just discard it.

### Time complexity of Carr-Madan
Let dimension be d. Then the complexity would be $O(N^d\log N)$, while the FDM complexity (assuming tridiagonal matrix) is $O(N^{2d})$.

# 10.15
Read "Multi-asset option pricing using a parallel Fourier-based technique (C. C. W. Leentvaar, C. W. Oosterlee)" multiple times, trying to implement it but failed.

# 10.22
Implemented Conv method for the 1d case based on "A fast and accurate FFT-based method for pricing early-exercise options under Levy processes (R. Lord et al). Derived the characteristic function for n-d GBM. In the process of extending the 1d case to the n-d case. 

### Why option prices at the middle of interval are far more accurate than ones at the end
Same idea as the Monte-Carlo method for highly out-of-money options.

# 10.29
Continue writing the thesis. Finish the writing of n-d case. Found a bug related to FFT and characteristic function in the thesis on Monday.

# 11.5
Identified and fixed the characteristic function bug and the change of variable bug in the thesis. Successfully implemented the n-d conv method and make it 100x faster by vectorization.

# 11.12
Add medium dimensional option pricing benchmark (geometric avg payoff). Add experiment module to simplify numerical experiments. Investigate in the reason why Conv method only gives solutions on domain close to the spot price: circular convolution. Give the analytical form of the greeks using Conv method.

# 11.19
Investigate more closely in the circular convolution. Understand Tensorflow in more depth and implemented the deep Galerkin method N-d European case. Wrote the first two chapters in the thesis: 1. Introduction and 2. Two Frameworks for Option Pricing. Experimented with N-d MC method using sobol sequence. Noticed that the N-d antithetic method may also be useful.

# 11.26
Thesis: Described three types of low discrepancy methods and their implementations. Completed the gap in "the error bound of the dgm" by proving the BS equation is a parabolic pde. Added the statement of maximum principle. Briefly introduced the architecture of dgm and bsde and compared the two methods.

# 12.3
Implemented antithetic variates for n-d MC and added the discussion of it to the paper. Figure out the reason why DGM using Hessian is extremely slow: batch size too large. Took error from the boundary conditions of the PDE into consideration.

# 12.10 and 12.17
Conducted numerical experiments and finished the thesis.