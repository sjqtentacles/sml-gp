structure Tests = struct open Harness structure G = Gp
fun run () = let
  (* --- single training point: closed-form GP mean is non-circular ---
     With one point (x0,y0), K = [[1+jitter]], alpha = y0/(1+jitter) ~ y0,
     so mean(xq) = y0 * rbf(l, xq, x0).  At x0 this is y0; at xq=1, l=1,
     x0=0, y0=2  ->  2 * exp(-0.5) = 1.213061... *)
  val () = section "single-point closed form"
  val p1 = G.predict [0.0] [2.0] [0.0, 1.0, 2.0] 1.0
  val () = checkRealTol 1E~6 "at training point" (2.0, List.nth (p1, 0))
  val () = checkRealTol 1E~6 "2*exp(-1/2)"      (2.0 * Math.exp (~0.5), List.nth (p1, 1))
  val () = checkRealTol 1E~6 "2*exp(-2)"        (2.0 * Math.exp (~2.0), List.nth (p1, 2))

  (* --- interpolation: at training inputs the posterior mean reproduces ys ---
     A plain kernel-weighted sum (the old stub) would NOT reproduce these,
     because neighbouring basis functions overlap; only inverting K does. *)
  val () = section "interpolates training data"
  val xs = [0.0, 1.0, 2.0, 3.0]
  val ys = [0.0, 1.0, 0.0, ~1.0]
  val pred = G.predict xs ys xs 1.0
  val () = checkRealTol 1E~5 "f(0)=0"  (0.0,  List.nth (pred, 0))
  val () = checkRealTol 1E~5 "f(1)=1"  (1.0,  List.nth (pred, 1))
  val () = checkRealTol 1E~5 "f(2)=0"  (0.0,  List.nth (pred, 2))
  val () = checkRealTol 1E~5 "f(3)=-1" (~1.0, List.nth (pred, 3))

  (* --- smoothness: a query between training points stays bounded by the
     surrounding targets (sanity, not a tautology) --- *)
  val () = section "interpolant stays bounded between knots"
  val mid = List.nth (G.predict xs ys [1.5] 1.0, 0)
  val () = check "0 <= f(1.5) <= 1" (mid >= ~0.01 andalso mid <= 1.01)
in Harness.run () end end
