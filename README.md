# sml-gp

[![CI](https://github.com/sjqtentacles/sml-gp/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-gp/actions/workflows/ci.yml)

Gaussian-process regression for Standard ML with a squared-exponential (RBF)
kernel. Computes the posterior **mean** prediction by solving the kernel linear
system — not a kernel-weighted average shortcut.

## API

```sml
Gp.rbf l a b            (* RBF kernel exp(-(a-b)^2 / (2 l^2)) *)
Gp.predict xs ys query l
```

`predict` forms the kernel matrix `K` over the training inputs `xs` (with a
small jitter on the diagonal for numerical stability), solves `K α = ys`, and
predicts each query point as `Σ αᵢ k(x*, xᵢ)`. With a noiseless kernel this
interpolates the training data exactly.

```sml
val xs = [0.0, 1.0, 2.0]
val ys = [0.0, 1.0, 0.0]
Gp.predict xs ys [0.0, 1.0, 2.0] 1.0   (* ~ [0.0, 1.0, 0.0] : reproduces training points *)
```

## Scope and limitations

- **Posterior mean only.** Predictive variance / covariance is not returned.
- Fixed RBF kernel and a fixed length-scale `l` passed per call. No kernel
  hyper-parameter optimization (no marginal-likelihood maximization).
- Noise is modelled only as a small fixed diagonal jitter; there is no tunable
  observation-noise term.
- Solves a dense `n×n` system via `sml-matrix`, so it is intended for modest
  training-set sizes.

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
  gp.sml       RBF kernel + GP regression (solves Kα = y)
  gp.mlb
test/
  test.sml     closed-form mean, interpolation, boundedness
```

## License

MIT. See [LICENSE](LICENSE).
