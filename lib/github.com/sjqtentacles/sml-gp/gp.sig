signature GP =
sig
  val rbf : real -> real -> real -> real
  val predict : real list -> real list -> real list -> real -> real list
end
