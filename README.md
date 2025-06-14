I will be periodically adding CUDA C/C++ implementations of common models in finance. 
Note, there are additional optimizations that can be made (calculating common terms on the CPU, multi-kernel iterations, etc.)

Current Files:
1. Heston: implementation of the Heston model (stochastic volatility). On a side note, I used the Euler method for discretization, which could allow for negative variance, which is impossible. Therefore, for each timestep, I check if variance is negative and if it is, I swap it out for 0.0. A better way of doing this would be to use a more sophisticated process to discretize the model (e.g quadratic exponential method)

2. Hull-White: implementation of the Hull-White model (short term rate). Like with the Heston model, I used the Euler method. The Hull-White model has an exact discretization, which is far more accurate, but the computation of the integral in this method can be complex. Perhaps I will make this change in a future iteration.

3. Vasicek: implementaion of the Vasicek model (short term rate). I used the exact discretization rather than the Euler method to eliminate discretization error. It is straightforward to calculate unlike with Hull-White. 

