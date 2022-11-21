# Sketch-and-solve approaches to k-means clustering by semidefinite programming
Authors: Charles Clum, Dustin G. Mixon, Soledad Villar, Kaiying Xie.

We introduce a sketch-and-solve approach to speed up the Peng-Wei semidefinite relaxation of k-means clustering. Our approach provides a high-confidence lower bound on the optimal k-means value. This lower bound is data-driven; it does not make any assumption on the data nor how it is generated.


## Implementation
This requires CVX [2] and SDPNAL+ [3].

- Use sketch_and_solve_lower_bound.m to obtain a high-confidence lower bound on the optimal k-means value for a given data matrix.

- main.m provides examples of how we use sketch_and_solve_lower_bound.m to obtain a lower bound for five different datasets. This prints out numerical results in Table 1 and Table 2 in [1].


## References
[1] C. Clum, D. G. Mixon, S. Villar, K. Xie, Sketch-and-solve approaches to k-means clustering by semidefinite programming.

[2] M. Grant, S. Boyd, CVX: Matlab software for disciplined convex programming, cvxr.com/cvx.

[3] D. F. Sun, K. C. Toh, Y. Yuan, X. Y. Zhao, SDPNAL+: A Matlab software for semidefinite programming with bound constraints (version 1.0), Optim. Methods Softw 35 (2020) 87-115.
