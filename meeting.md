# 7.31
### Cholesky decomposition for correlation matrix
Corr = $\Rho = LL^T$. $dW = LN\sqrt{dt}$ where $N$ is a normal distribution RV vector. In such casem we will have $$corr(dW_i, dW_j) = E[dw_idw_j]-E[dw_i]E[dw_j] = E[dw_idw_j]-0 = \Rho_{ij}$$

### Feyman-Kac can transform second order linear pde into bsde

### My other questions during meeting seem to be too trivial now (8.23). Thus skipped.
# 8.7
### Complete delta heding
During the delta hedging process, we need Delta at each time step for the current stock price. We can retrieve that either by running the MC again, or interpolate using results gained from previous MC.

### My other questions during meeting seem to be too trivial now (8.23). Thus skipped.
# 8.14
### F1 score
F1 score is a good indicator of accuracy when the proportion of false positive and true negative is skewed.
$$F_1 = 2\frac{precision\cdot recall}{precision+recall}$$
where $precision = \frac{tp}{tp+fp}$, $recall=\frac{tp}{tp+fn}$.

### TODO: P&L why not centre at x=1?

### Use delta hedging to verify the accuracy of Delta?

### Figured out why Chen's multi-step architecture is more efficient
There are less neural networks in a single pricing model.

# 8.23
### Overfitting of LSMC by choosing high order monomial basis
We need to guarantee that the basis # is much smaller than the sample size. It's also recommended to add a regularization term:
Initially we want $min_c ||Ac- y||_2^2$, now we add a regularization term $\rho||c||_2^2$. (Note that sometimes we use 1-norm for the regularization term, in which case we have to solve it using iterative methods.) So it becomes
$$min_c (||Ac- y||_2^2+\rho||c||_2^2) $$
$$=min_c(||\Big[A\ \ I\Big]^Tc - \Big[y \ \ 0\Big]^T||)$$
which again can be solved by QR or SVD.
The analytical form of original sol is $(A^TA)c = A^Ty$.

### It's more accurate to simulate stock price by a kinda analytical sol instead of the crude Euler method
$$S_t = S_0e^{(r-\sigma^2/2)t + \sigma\sqrt {t} z}, where\ Z\sim N(0,1).$$
European option pricing can be accelerated greatly by this, since we can directly get the price at time T.
##### TODO: Generalize it into multi-asset version.

### Variance decomposition of antithetic variates
In P208 Glasserman, we decompose $f(z)$ into an even function $f_0(z) = \frac{f(z)+f(-z)}{2}$ and an odd function $f_1(z) = \frac{f(z)-f(-z)}{2}$. We show that $f_0$ and $f_1$ are uncorrelated:
$$E[f_0(Z)f_1(Z)] = \frac{1}{4}E[f^2(Z)-f^2(-Z)]$$
Let the pdf of Z be $p(z)$,
$$E[f^2(Z)] = \int_{-\infty}^\infty f^2(x)p(x)dx = -\int_{\infty}^{-\infty} f^2(-x)p(-x)dx = \int_{-\infty}^\infty f^2(-x)p(-x)dx=E[f^2(-Z)] $$
So
$$E[f_0(Z)f_1(Z)] = 0.$$
It follows that 
$$Var[f(Z)] = Var[f_0(Z)] + Var[f_1(Z)].$$
The first term on the right is the variance of an estimate of $E[f(Z)]$ based on an antithetic pair $(Z, -Z)$. So if $f$ is very odd ($||f-f_1||$ is small), the antithetic variates would work well. Vice versa.


### PCA covariance matrix
In the usual problem, the covariance matrix needs to be approximated. But in option pricing, we assume that a correlation matrix is given. In this case, we can recover the correlation matrix to a covariance matrix by only approximating variance of stock price (TODO: Can be analytically expressed?). It should give a more accurate price.
