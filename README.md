# krylov4r
*Krylov methods for research*

The krylov4r Matlab package offers Krylov algorithms implementations for researchers, who require more options and outputs than the functions natively provided by Matlab. Currently, GMRES and GCR are implemented. The following features are available:
- [x] left and right preconditioners
- [x] restart
- [x] deflation spaces
- [x] change of inner product
- [x] choose the orthogonalization algorithm (Gram-Schmidt, Modified Gram-Schmidt) and the number of re-orthogonalizations
- [x] choose how the residual is computed (preconditioned or not, in weighted norm or not)
- [x] set breakdown tolerance
- [x] outputs absolute and relative residual norms
- [x] outputs successive solution vectors

## Install
Download the repository and add `<path>/krylov4r/src/` to your Matlab/Octave path.

## Getting started

```matlab
[x, flag, relres, iter, absresvec, relresvec, xvec] =
      gmres4r(A, b, 'left_prec', HL, 'right_prec', HR, 'restart', 5, 'tol', 1e-8);
```
Except for `A,b`, all parameters are optional, named, and can be set in any order.

For documentation, run
```matlab
help gmres4r
help gcr4r
```
