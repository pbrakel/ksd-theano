(KSDT) Krylov Subspace Descent using Theano.
============================================

This is an implementation of the Krylov Subspace Descent method described in
the AISTATS paper by Vinyals and Povey (2012). The code is loosely based on an implementation of the
Hessian-Free optimizer that can be found at https://github.com/boulanni/theano-hf.

The algorithm is implemented using Theano for the parts that are the most
compuationally intensive. I implemented both the preconditioner by James
Martens and the Jacobi pre-conditioner.

Krylov Subspace Descent uses repeated matrix vector multiplications to estimate
an orthonormal basis for what is called a Krylov Subspace. The method can be
seen as a dimensionality reduction method that finds usefull directions in the
parameter space. Once the directions have been computed, they are searched over
using the BFGS implementation from Scipy. A nice thing about this optimizer is
that is requires very little hyper parameter tuning except for the number of
search directions to compute and the maximum number of BFGS iterations to
perform.

Please let me know if you find any bugs or have any questions about this code.
