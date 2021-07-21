# QP derivatives

Computes the derivatives of the solution of a quadratic program (QP) w.r.t. the QP's parameters using 
- Numerical differentiation,
- Sensitivity analysis (Inverse Function Theorem). 

Notations in the code are adapted from [OptNet](https://arxiv.org/abs/1703.00443). At the moment the QP does not have equality constraints. It can be written as
```
min. 1 / 2 * z @ Q @ z + b @ z
s.t. G @ z <= e
```
