# pytope
Package with a limited set of operations for polytopes, zonotopes, and invariant sets.

The currently implemented features include 
* constructing polytopes from inequalities (halfspace representation), vertices, and upper and lower bounds, 
* linear mapping (multiplying a matrix *M* and a polytope *P*: *M P*), 
* the Minkowski sum and Pontryagin difference of polytopes, 
* the intersection of polytopes, and
* simple plotting. 

The figures below are generated with pytope. 
The first two illustrate some of the currently implemented operations (see [demo.py](https://github.com/heirung/pytope/blob/master/pytope/demo.py)); 
the third uses pytope to plot a robust MPC trajectory (obtained with [CasADi](https://github.com/casadi)), combining Figures 1 and 2 from Mayne et al. (2005) with a rough approximation of the disturbance sequence; 
the fourth shows an outer *&epsilon;*-approximation of a minimal robust positively invariant, or MRPI, set computed with pytope, reproducing the example from Raković et al. (2005).  

pytope is experimental, fragile, largely untested, and probably buggy.

![Illustration of various polytope operations](https://raw.githubusercontent.com/heirung/pytope/master/docs/various_operations.svg?sanitize=true)  
Figure: Illustration of various polytope operations.

![The Minkowski sum of two polytopes](https://raw.githubusercontent.com/heirung/pytope/master/docs/minkowski_sum.svg?sanitize=true)  
Figure: The Minkowski sum of two polytopes.

![Robust MPC trajectory from Mayne et al. (2005)](https://raw.githubusercontent.com/heirung/pytope/master/docs/Mayne_2005.svg?sanitize=true)  
Figure: Robust MPC trajectory from Mayne et al. (2005), combining Figures 1 and 2.

![Outer *&epsilon;*-approximation of a minimal RPI](https://raw.githubusercontent.com/heirung/pytope/master/docs/Rakovic_2005.svg?sanitize=true)  
Figure: Outer *&epsilon;*-approximation of a minimal robust positively invariant (MRPI) set – a reproduction of the example in Raković et al. (2005).

### References
* Mayne, D.Q., Seron, M.M., & Raković, S.V. (2005). 
Robust model predictive control of constrained linear systems with bounded disturbances. 
[*Automatica*, 41(2), 219–224](https://doi.org/10.1016/j.automatica.2004.08.019).
* Raković, S.V., Kerrigan, E.C., Kouramas, K.I., & Mayne, D.Q. (2005). 
Invariant approximations of the minimal robust positively invariant set. 
[*IEEE Transactions on Automatic Control*, 50(3), 406–410](https://doi.org/10.1109/TAC.2005.843854).