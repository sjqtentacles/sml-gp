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

  (* --- posterior variance: at a (near) noise-free training point the latent
     variance collapses to ~0; far from data it approaches the prior k(x,x)=1 --- *)
  val () = section "predictVar: variance shrinks at data, grows away"
  val kern = G.rbf 1.0
  val pv = G.predictVar kern 1.0E~8 xs ys [1.0, 100.0]
  val atData = List.nth (pv, 0)
  val farAway = List.nth (pv, 1)
  (* mean at training point ~ y(1) = 1 *)
  val () = checkRealTol 1E~4 "mean at x=1" (1.0, #mean atData)
  val () = check "var at data ~ 0" (#var atData >= ~1E~6 andalso #var atData <= 1E~3)
  val () = checkRealTol 1E~3 "var far away ~ prior 1" (1.0, #var farAway)

  (* --- mean from predictVar matches the legacy predict (same kernel, tiny noise) --- *)
  val () = section "predictVar mean agrees with predict"
  val q = [0.5, 1.5, 2.5]
  val m1 = G.predict xs ys q 1.0
  val m2 = List.map #mean (G.predictVar kern 1.0E~9 xs ys q)
  val () = checkRealTol 1E~4 "q0" (List.nth (m1,0), List.nth (m2,0))
  val () = checkRealTol 1E~4 "q1" (List.nth (m1,1), List.nth (m2,1))
  val () = checkRealTol 1E~4 "q2" (List.nth (m1,2), List.nth (m2,2))

  (* --- log marginal likelihood: finite, and a length-scale that fits the data
     better gives a higher (less negative) value than a clearly wrong one --- *)
  val () = section "logMarginalLikelihood"
  val lmlGood = G.logMarginalLikelihood (G.rbf 1.0) 1.0E~2 xs ys
  val lmlBad  = G.logMarginalLikelihood (G.rbf 0.01) 1.0E~2 xs ys
  val () = check "lml is finite" (Real.isFinite lmlGood)
  val () = check "better length-scale -> higher lml" (lmlGood > lmlBad)

  (* --- matern32 is a valid kernel: 1 at zero distance, decreasing, positive --- *)
  val () = section "matern32 kernel shape"
  val mk = G.matern32 1.0
  val () = checkRealTol 1E~9 "k(x,x)=1" (1.0, mk 2.0 2.0)
  val () = check "k decreases with distance" (mk 0.0 1.0 > mk 0.0 2.0)
  val () = check "k stays positive" (mk 0.0 5.0 > 0.0)
  (* GP with matern kernel still (near) interpolates noise-free data *)
  val pm = G.predictVar mk 1.0E~8 xs ys xs
  val () = checkRealTol 1E~3 "matern mean f(1)=1" (1.0, #mean (List.nth (pm, 1)))
in Harness.run () end end
