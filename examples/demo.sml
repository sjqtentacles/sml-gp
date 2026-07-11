(* demo.sml - Gaussian process regression over five noisy samples of sin(x):
   RBF and Matern-3/2 kernels, posterior mean, posterior mean+variance, and
   the log marginal likelihood. Deterministic: fixed training/query points,
   no wall clock, no randomness. Identical output on every run and both
   compilers. *)

structure G = Gp

fun fmtR n r =
  let val r = if Real.== (r, 0.0) then 0.0 else r
  in Real.fmt (StringCvt.FIX (SOME n)) r end
fun fmtRList n xs = "[" ^ String.concatWith "," (List.map (fmtR n) xs) ^ "]"

val () = print "GP demo\n"

(* Five samples of sin(x) on [0,4]. *)
val xs = [0.0, 1.0, 2.0, 3.0, 4.0]
val ys = [0.0, 0.84, 0.91, 0.14, ~0.76]
val queries = [0.5, 1.5, 2.5, 3.5]

val rbf = G.rbf 1.0
val matern = G.matern32 1.0
val () = print ("rbf 1.0: k(0,0)=" ^ fmtR 4 (rbf 0.0 0.0) ^ ", k(0,1)=" ^ fmtR 4 (rbf 0.0 1.0) ^ "\n")
val () = print ("matern32 1.0: k(0,0)=" ^ fmtR 4 (matern 0.0 0.0) ^ ", k(0,1)=" ^ fmtR 4 (matern 0.0 1.0) ^ "\n")

val meanOnly = G.predict xs ys queries 1.0
val () = print ("predict (RBF, l=1.0) at " ^ fmtRList 1 queries ^ "\n")
val () = print ("  posterior mean = " ^ fmtRList 4 meanOnly ^ "\n")

val withVar = G.predictVar rbf 0.01 xs ys queries
val () =
  List.app
    (fn (q, { mean, var }) =>
      print ("  x=" ^ fmtR 1 q ^ ": mean=" ^ fmtR 4 mean ^ ", var=" ^ fmtR 4 var ^ "\n"))
    (ListPair.zip (queries, withVar))

val lml = G.logMarginalLikelihood rbf 0.01 xs ys
val () = print ("log marginal likelihood (RBF, sigma^2=0.01) = " ^ fmtR 4 lml ^ "\n")

val lmlMatern = G.logMarginalLikelihood matern 0.01 xs ys
val () = print ("log marginal likelihood (Matern32, sigma^2=0.01) = " ^ fmtR 4 lmlMatern ^ "\n")
