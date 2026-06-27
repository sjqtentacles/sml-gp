signature GP =
sig
  (* A covariance kernel: k x x' -> covariance. *)
  type kernel = real -> real -> real

  (* Squared-exponential (RBF) kernel with length-scale l. *)
  val rbf : real -> kernel

  (* Matern 3/2 kernel with length-scale l:
        k(x,x') = (1 + sqrt 3 r / l) * exp(- sqrt 3 r / l),  r = |x - x'|. *)
  val matern32 : real -> kernel

  (* Posterior-mean GP regression with the RBF kernel and length-scale l
     (unchanged from the original API; uses a tiny fixed jitter). *)
  val predict : real list -> real list -> real list -> real -> real list

  (* Posterior mean AND latent variance at each query point, for an arbitrary
     kernel and an explicit observation-noise variance (added to the diagonal of
     the training covariance). Variance is k(xq,xq) - k_*^T (K+sigma^2 I)^-1 k_*,
     i.e. the variance of the latent function (predictive-noise not added). *)
  val predictVar : kernel -> real          (* noise variance sigma^2 *)
                 -> real list -> real list  (* training xs, ys *)
                 -> real list               (* query points *)
                 -> { mean : real, var : real } list

  (* Log marginal likelihood log p(y | X) under the GP prior with the given
     kernel and noise variance, computed stably from the Cholesky factor:
        -1/2 y^T (K+sI)^-1 y  -  sum_i log L_ii  -  n/2 log (2 pi). *)
  val logMarginalLikelihood : kernel -> real -> real list -> real list -> real
end
