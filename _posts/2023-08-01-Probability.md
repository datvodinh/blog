---
title: "Probability For Machine Learning (Part 1)"
date: 2023-06-10 09:00:00 +0700
categories: [Machine Learning, AI]
tags: [ml, ai, nlp, rnn] # TAG names should always be lowercase
img_path: /assets/img/Prob/
math: true
image:
  path: prob.jpg
  width: 300
  height: 600
  alt: Probability
---

## Random variables, distributions, and moments

### Random variables

- A _random variable_ is a function which assigns a number to _events_ in the _sample space_. (A better name might be "random-valued function on the sample space.")

$$X = \{1:heads,0:tail\}$$

- We describe the _probability of an outcome_ in terms of the probability of a random variable taking a given value:

$$P(X = 1) = 1/2, P(X^2 = 1) = 1$$

### Continuous random variables

- Consider choosing a random number between 0 and 1, where all values are equally likely.
- Since there are an (uncountably) _infinite_ number of values, the probability of any given value is zero:$$Prob(X = x_0) = 0$$
- Does not mean event is impossible. Points on a line have "_measure zero_"
- Instead, ask for probability to lie within a given range, e.g.,

$$Prob(a < X<b) = \int_a^b p(x)dx$$

### Probability distributions

- More generally, a probability distribution satisfies:
  - $P(x) \geq 0$
  - $\sum_k p(x_k) = 1, x_k$ discrete
  - $\int_{-\infty}^{\infty}p(x)dx=1, x$ continuous

### Cumulative distribution function

- The _cumulative distribution function_, or CDF, gives the probability that X is less than or equal to a given value.

$$F(x) = Prob(X<x) = \int_{-\infty}^x p(x^{'})dx^{'}$$

- It contains (nearly) the same information as the probability density, since

$$p(x) = \frac{dF(x)}{dx}$$

- The CDF is often easier to approximate from empirical data and it is useful since

$$Prob(a < X < b) = F(b) - F(a)$$

### Change of variable

- Suppose we wish to change variables, shift the distribution, or consider functions of the random variable. If $x \rightarrow y = y(x)$ then the density in terms of the new variable is given by

$$p(x)dx = g(y)dy$$

which preserves the normalization condition. (This assumes y is an increasing function of x. If not, an absolute value is needed for positivity; and further care is needed if y has critical points.)

$$p(x)dx = g(y)dy$$
$$g(y) = \frac{p(x)}{|dy / dx|}$$

### Expectations and moments

- The probability distribution defines weighted averages over the sample space, where the weight of each event is equal to its probability. These are called _expected values_.

- For the discrete case,

$$E[f(X)] = \sum_{k=1}^n f(x_k)p(x_k)$$

- while for the continuous case,

$$E[f] = \int_{-\infty}^{\infty}f(x)p(x)dx$$

### Mean of a distribution

- The _mean_ of the distribution is simply the expectation of the random variable itself:

$$\mu = E[X] = \overline{X} = \langle X \rangle = \left\{ \begin{array}{rcl} \sum_kx_kp(x_k) \\ \int xp(x)dx \end{array}\right.$$

- In the case of an infinite sample space, whether continuous or discrete, the mean is not guaranteed to exist since the integral or the sum might not converge.

### Moments of a distribution

- The _moments_ of a distribution are the expectation of _powers_ of the random variable itself.

$$\mu_l \equiv E[X^l]  \equiv \langle X^l \rangle = \left\{ \begin{array}{rcl} \sum_kx_k^lp(x_k) \\ \int x^lp(x)dx \end{array}\right.$$

- If all the moments are known – and if they exist – they can be used to get the expectation of other functions using the _linearity_ of the expectation operator

$$E[cf(X)] = cE[f(x)]$$

$$E[f(x) + g(X)] = E[f(x)] + E[g(x)]$$

### Variance and standard deviation

- Of particular interest is the second moment, in combination with the mean, defining the variance:

$$\sigma^2 = Var(X) = E[(X-\mu)^2] = E[X^2] - E[X]^2$$

- The standard deviation, which is the square root of the variance, has the _same units_ as the random variable (e.g., rate of return, dollars, etc.)

### Higher moments characterize properties of a distribution

- _Variance_ – dispersion measure based on second moment

$$\sigma^2 \equiv E[(X-\mu)^2] = \int (x-\mu)^2p(x)dx$$

- _Skewness_ – asymmetry parameter based on 3rd moments; dimensionless – normalized cumulant

$$s \equiv \frac{E[(X-\mu)^3]}{\sigma^3} = E[(\frac{X-\mu}{\sigma})^3]$$

- _Kurtosis_ – measure of tail "weights" in terms of 4th moments; zero for Gaussian, bounded below by -1.

$$\kappa \equiv \frac{E[(X-\mu)^4]}{\sigma^4} - 3$$

### Covariance and correlation

- For any _two_ random variables, not necessarily independent or identically distributed, their covariance is defined as

$$Cov(X,Y) \equiv E[(X - \mu_x)(Y - \mu_y)] = E[XY] - \mu_x \mu_y$$

- The correlation is proportional to the covariance,

$$\rho(X,Y) = Corr(X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}} = E[(\frac{X-\mu_x}{\sigma_x})(\frac{Y - \mu_y}{\sigma_y})]$$

- Dividing the covariance by the standard deviations makes the correlation a pure number, and

$$-1 \leq \rho(X,Y) \leq +1$$

### Summary

- "_Random variables_" are functions that assign a _number_ to events in the sample space. They can be discrete, continuous, or a mix of both.
- The _probability distribution_ is _positive_ and _sums to one_.
- Expected values, or expectations, are weighted averages that use the probabilities as the weights.
- The _moments_ of a distribution, such as the mean, are expectations of various powers of a random variable. They are numbers, not functions.
- _Variance_, _skewness_, and _kurtosis_ are simple functions of the moments that characterize the shape of the probability distribution.
- When there are multiple random variables, their _covariance_ and _correlation_ are also computed as expectations.

(To be Continued!)
