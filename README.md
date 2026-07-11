# sml-gp

[![CI](https://github.com/sjqtentacles/sml-gp/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-gp/actions/workflows/ci.yml)

Gaussian-process regression for Standard ML with squared-exponential (RBF) and
Matern-3/2 kernels. Computes the posterior **mean and variance** by solving the
kernel linear system — not a kernel-weighted average shortcut — and the log
marginal likelihood from a Cholesky factor.

## API

```sml
type kernel = real -> real -> real

Gp.rbf l                 (* RBF kernel  exp(-(a-b)^2 / (2 l^2)) *)
Gp.matern32 l            (* Matern 3/2 kernel *)

(* posterior mean (legacy, noise-free, RBF) *)
Gp.predict xs ys query l

(* posterior mean AND latent variance, arbitrary kernel + noise variance *)
Gp.predictVar kernel noise xs ys query   (* -> { mean, var } list *)

(* log marginal likelihood log p(y | X) via Cholesky *)
Gp.logMarginalLikelihood kernel noise xs ys
```

`predict`/`predictVar` form the kernel matrix `K` over the training inputs `xs`
(plus a noise term on the diagonal), solve `(K+σ²I) α = ys`, and predict each
query point as `Σ αᵢ k(x*, xᵢ)`. The latent variance is
`k(x*,x*) − k_*ᵀ (K+σ²I)⁻¹ k_*`, which collapses toward zero near observed
points and rises to the prior variance far from data.

```sml
val xs = [0.0, 1.0, 2.0]
val ys = [0.0, 1.0, 0.0]
Gp.predict xs ys [0.0, 1.0, 2.0] 1.0   (* ~ [0.0, 1.0, 0.0] : reproduces training points *)

val k = Gp.rbf 1.0
Gp.predictVar k 1.0E~6 xs ys [1.0, 100.0]
(* [ {mean ≈ 1.0, var ≈ 0.0}, {mean ≈ 0.0, var ≈ 1.0} ] *)

(* pick a length-scale by comparing marginal likelihoods *)
Gp.logMarginalLikelihood (Gp.rbf 1.0)  0.01 xs ys   (* > *)
Gp.logMarginalLikelihood (Gp.rbf 0.01) 0.01 xs ys   (* worse fit *)
```

## Scope and limitations

- **Manual hyper-parameters.** `logMarginalLikelihood` lets you compare
  kernel/length-scale/noise choices, but there is no built-in optimizer; you
  evaluate candidates yourself.
- `predictVar` returns the **latent** function variance (it does not add the
  observation-noise variance back for a predictive interval — add `noise` if you
  want the noisy-observation variance).
- Solves a dense `n×n` system via `sml-matrix`, so it is intended for modest
  training-set sizes.

## Example

`make example` builds and runs [`examples/demo.sml`](examples/demo.sml), which
fits both kernels to five noisy samples of sin(x), computes the posterior
mean and mean+variance at four query points, and the log marginal likelihood
under each kernel (output is byte-identical under MLton and Poly/ML):

```
GP demo
rbf 1.0: k(0,0)=1.0000, k(0,1)=0.6065
matern32 1.0: k(0,0)=1.0000, k(0,1)=0.4834
predict (RBF, l=1.0) at [0.5,1.5,2.5,3.5]
  posterior mean = [0.4383,1.0077,0.6049,~0.3834]
  x=0.5: mean=0.4324, var=0.0221
  x=1.5: mean=1.0016, var=0.0160
  x=2.5: mean=0.6016, var=0.0160
  x=3.5: mean=~0.3839, var=0.0221
log marginal likelihood (RBF, sigma^2=0.01) = ~4.4706
log marginal likelihood (Matern32, sigma^2=0.01) = ~5.0609
```

## Installing with smlpkg

```sh
smlpkg add github.com/sjqtentacles/sml-gp
smlpkg sync
```

Reference from your `.mlb`:

```
lib/github.com/sjqtentacles/sml-gp/gp.mlb
```

## Building and testing

```sh
make test        # MLton
make test-poly   # Poly/ML
make all-tests   # both
make clean
```

## Project layout

```
sml.pkg
Makefile
lib/github.com/sjqtentacles/sml-gp/
  gp.sig
  gp.sml       RBF + Matern kernels, GP mean/variance, log marginal likelihood
  gp.mlb
test/
  test.sml     closed-form mean, interpolation, variance, logML, matern
```

## License

MIT. See [LICENSE](LICENSE).
