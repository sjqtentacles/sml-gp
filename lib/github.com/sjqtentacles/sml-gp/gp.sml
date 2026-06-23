structure Gp :> GP =
struct
  (* Squared-exponential (RBF) kernel with length-scale l. *)
  fun rbf l x x' = Math.exp (~0.5 * Math.pow ((x - x') / l, 2.0))

  (* Tiny diagonal jitter: regularises K so the linear solve is stable and the
     posterior mean very nearly interpolates the (noise-free) training data. *)
  val jitter = 1.0E~9

  (* Gaussian-process posterior mean regression.

     Given training inputs xs, targets ys and length-scale l, this solves the
     GP normal equations  (K + jitter*I) alpha = ys  for the dual weights
     alpha (K_ij = rbf(x_i, x_j)), then predicts at each query point xq as
        mean(xq) = sum_i alpha_i * rbf(xq, x_i).
     This is the genuine kernel-ridge / GP-mean computation (it inverts the
     kernel matrix); it is NOT a plain kernel-weighted average. *)
  fun predict xs ys query l =
    let
      val n = List.length xs
    in
      if n = 0 then List.map (fn _ => 0.0) query
      else
        let
          val xv = Vector.fromList xs
          val rowsK =
            List.tabulate (n, fn i =>
              List.tabulate (n, fn j =>
                rbf l (Vector.sub (xv, i)) (Vector.sub (xv, j))
                + (if i = j then jitter else 0.0)))
          val k     = Matrix.fromRows rowsK
          val alpha = Vector.fromList (Matrix.solve k ys)
          fun meanAt xq =
            let
              fun acc (i, s) =
                if i >= n then s
                else acc (i + 1, s + Vector.sub (alpha, i) * rbf l xq (Vector.sub (xv, i)))
            in acc (0, 0.0) end
        in
          List.map meanAt query
        end
    end
end
