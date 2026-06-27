structure Gp :> GP =
struct
  type kernel = real -> real -> real

  (* Squared-exponential (RBF) kernel with length-scale l. *)
  fun rbf l x x' = Math.exp (~0.5 * Math.pow ((x - x') / l, 2.0))

  (* Matern 3/2 kernel with length-scale l. *)
  fun matern32 l x x' =
    let val r = Real.abs (x - x')
        val s = Math.sqrt 3.0 * r / l
    in (1.0 + s) * Math.exp (~s) end

  (* Tiny diagonal jitter for the legacy noise-free predict. *)
  val jitter = 1.0E~9

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

  (* Build (K + noise*I) for the training inputs under kernel k. *)
  fun gram (k : kernel) noise (xv : real vector) =
    let val n = Vector.length xv in
      Matrix.fromRows
        (List.tabulate (n, fn i =>
          List.tabulate (n, fn j =>
            k (Vector.sub (xv, i)) (Vector.sub (xv, j))
            + (if i = j then noise else 0.0))))
    end

  fun dot (a, b) =
    let val n = List.length a
        val av = Vector.fromList a
        val bv = Vector.fromList b
        fun go (i, s) = if i >= n then s
                        else go (i + 1, s + Vector.sub (av, i) * Vector.sub (bv, i))
    in go (0, 0.0) end

  fun predictVar (k : kernel) noise xs ys query =
    let
      val n = List.length xs
    in
      if n = 0 then List.map (fn xq => { mean = 0.0, var = k xq xq }) query
      else
        let
          val xv = Vector.fromList xs
          val ky = gram k noise xv
          val alpha = Matrix.solve ky ys             (* (K+sI)^-1 y *)
          val kinv = Matrix.inv ky
          fun kstar xq = List.tabulate (n, fn i => k xq (Vector.sub (xv, i)))
          fun meanAt (ks, _) = dot (alpha, ks)
          fun varAt (ks, xq) =
            let
              (* v = Kinv * ks  (length n) *)
              val ksv = Vector.fromList ks
              fun row i =
                let fun go (j, s) = if j >= n then s
                          else go (j + 1, s + Matrix.sub (kinv, i, j) * Vector.sub (ksv, j))
                in go (0, 0.0) end
              val v = List.tabulate (n, row)
            in k xq xq - dot (ks, v) end
          fun at xq = let val ks = kstar xq
                      in { mean = meanAt (ks, xq), var = varAt (ks, xq) } end
        in List.map at query end
    end

  fun logMarginalLikelihood (k : kernel) noise xs ys =
    let
      val n = List.length xs
    in
      if n = 0 then 0.0
      else
        let
          val xv = Vector.fromList xs
          val ky = gram k noise xv
          val l = Matrix.cholesky ky            (* lower-triangular, K = L Lᵀ *)
          val alpha = Matrix.solve ky ys        (* (K+sI)^-1 y *)
          val quad = dot (ys, alpha)            (* yᵀ (K+sI)^-1 y *)
          (* log|K| = 2 * sum log L_ii *)
          fun sumLogDiag (i, acc) =
            if i >= n then acc
            else sumLogDiag (i + 1, acc + Math.ln (Matrix.sub (l, i, i)))
          val logDet = 2.0 * sumLogDiag (0, 0.0)
        in
          ~0.5 * quad - 0.5 * logDet - 0.5 * real n * Math.ln (2.0 * Math.pi)
        end
    end
end
