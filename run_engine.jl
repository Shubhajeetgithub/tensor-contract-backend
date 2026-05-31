include("TensorOps.jl")
using .TensorOps

# # Define a 4D Minkowski metric
# minkowski_data = [
#     -1.0  0.0  0.0  0.0;
#      0.0  1.0  0.0  0.0;
#      0.0  0.0  1.0  0.0;
#      0.0  0.0  0.0  1.0
# ]
# g = Tensor(minkowski_data, (:lower, :lower))

# # Define a 4-velocity vector (time-like)
# v_data = [1.0, 0.0, 0.0, 0.0]
# u = Tensor(v_data, (:upper,))

# # Perform the contraction to get the covector
# u_lowered = contract(g, u)
# println(minkowski_data)
# println("v_data =", v_data)
# println("u_lowered =", u_lowered)

# using Symbolics

# # 1. Define your coordinate variables symbolically
# @variables t r θ ϕ M

# # 2. Construct a symbolic data array (e.g., Schwarzschild Metric components)
# # Instead of Float64, this array holds 'Num' types
# schwarzschild_data = zeros(Num, 4, 4)
# schwarzschild_data[1, 1] = -(1 - 2M/r)  # g_tt
# schwarzschild_data[2, 2] = 1 / (1 - 2M/r) # g_rr
# schwarzschild_data[3, 3] = r^2          # g_θθ
# schwarzschild_data[4, 4] = r^2 * sin(θ)^2 # g_ϕϕ

# # 3. Wrap it in your custom Tensor class
# # Julia automatically infers T as 'Num' and N as 2
# g = Tensor(schwarzschild_data, (:lower, :lower))

# # 4. Perform a manual derivative loop (e.g., for Christoffel symbols)
# # Symbolics.differential handles analytical derivatives cleanly
# D_r = Differential(r)
# dg_dr = zeros(Num, 4, 4)

# for j in 1:4
#     for i in 1:4
#         # Take the analytical derivative component by component
#         dg_dr[i, j] = D_r(g.data[i, j]) |> expand_derivatives
#     end
# end

# # 5. Simplify the algebraic mess at the end
# simplified_dg_dr = Symbolics.simplify.(dg_dr)
# println(simplified_dg_dr)


minkowski_data = [
    -1.0 0.0 0.0 0.0;
     0.0 1.0 0.0 0.0;
     0.0 0.0 1.0 0.0;
     0.0 0.0 0.0 1.0
]
R_data = [
    2.0 1.0 0.0 -0.5;
    0.0 1.5 2.5  1.0;
   -3.0 0.0 1.0  0.0;
    1.0 2.0 -1.0  0.0
]
# A rank-4 Riemann-like tensor R^ρ_μ_σ_ν (4×4×4×4) for the self-contraction demo.
R4_raw = [R_data[a,b] * R_data[c,d]
          for a in 1:4, b in 1:4, c in 1:4, d in 1:4]
R4_data = R4_raw .- permutedims(R4_raw, (3,2,1,4))

g  = Tensor(minkowski_data, [lower(:μ), lower(:ν)], :g)
R  = Tensor(R_data, [upper(:μ), upper(:ν)], :R)
R4 = Tensor(R4_data, [upper(:ρ), lower(:μ), lower(:σ), lower(:ν)], :R4)
v  = Tensor([1.0, 2.0, 1.0, 3.0], [upper(:μ)], :v)
ns = Dict(:g => g, :v => v, :R => R, :R4 => R4)

println("Example 1:  v_ν = g_{μν} v^μ   (lower a vector)")
r1 = tensor_assign("v_ν = g{_μ _ν} v{^μ}", ns)
println(r1)

println()
println("Example 2:  A = g_{αβ} R^{αβ}  (double contraction / scalar)")
println("Note: labels α,β are mapped positionally to g's axes μ,ν")
r2 = tensor_assign("A = g{_α _β} R{^α ^β}", ns)
println(r2)

println()
println("Example 3:  R_{μν} = R4^ρ_{μρν}  (Ricci-like contraction)")
println("Single tensor, self-contracts over ρ")
r3 = tensor_assign("Ric_{μν} = R4{^ρ _μ _ρ _ν}", ns)
println(r3)

println()
println("Example 4:  Primed indices — Lorentz-transform style")
println("  Λ^{μ'}_{ν}  contracts  v^{ν}")
β, γ = 0.6, 1.25
Λ_data = [
     γ    -γ*β  0.0  0.0;
    -γ*β   γ    0.0  0.0;
     0.0   0.0  1.0  0.0;
     0.0   0.0  0.0  1.0
]
Λ = Tensor(Λ_data, [upper(Symbol("μ'")), lower(:ν)], :Λ)
push!(ns, :Λ => Λ)

r4 = tensor_assign("v'_{μ'} = Λ{^μ' _ν} v{^ν}", ns)
println(r4)