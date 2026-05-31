# Computes Christoffel symbols, Riemann / Ricci / Ricci-scalar / Einstein tensors.
# USAGE:
#   include("TensorOps.jl")
#   include("StandardTensors.jl")
#   using .StandardTensors
#   g_fn = x -> [... metric matrix ...]   # function R^n -> R^{n×n}
#   x0   = [0.0, 1.0, π/4, 0.0]          # coordinate point to evaluate at
#   Γ    = christoffel_symbols(g_fn, x0)   # Tensor Γ^λ_{μν}
#   Riem = riemann_tensor(g_fn, x0)        # Tensor R^ρ_{σμν}
#   Ric  = ricci_tensor(g_fn, x0)          # Tensor R_{μν}
#   Rs   = ricci_scalar(g_fn, x0)          # Float64
#   G    = einstein_tensor(g_fn, x0)       # Tensor G_{μν}

module StandardTensors
using ..TensorOps: Tensor, TensorIndex, TensorKind,
                  upper, lower, validate_indices,
                  contract, self_contract, rename_indices,
                  _contract_loops!, _self_contract_loops!

export metric_inverse,
       christoffel_symbols,
       riemann_tensor,
       ricci_tensor,
       ricci_scalar,
       einstein_tensor

# g_{μν}  →  g^{μν}
function metric_inverse(g_mat::AbstractMatrix)
    n = size(g_mat, 1)
    size(g_mat, 2) == n || error("Metric must be square")
    Tensor(inv(g_mat), [upper(:μ), upper(:ν)], :g_inv)
end
function metric_inverse(g::Tensor)
    ndims(g.data) == 2 || error("Metric tensor must be rank-2")
    metric_inverse(g.data)
end

# ── numerical derivatives ─────────────────────────────────────────────────────
function _d_scalar(f, x::Vector{Float64}, k::Int, h::Float64=1e-5)
    xp = copy(x); xp[k] += h
    xm = copy(x); xm[k] -= h
    (f(xp) - f(xm)) / (2h)
end

# ∂g_{μν}/∂x^σ at point x.  Returns a 3-index array dg[σ,μ,ν].
function _metric_partials(g_fn, x::Vector{Float64}, h::Float64=1e-5)
    n = length(x)
    g0 = g_fn(x)
    size(g0) == (n, n) || error("g_fn must return an $n×$n matrix at a length-$n coordinate vector")
    dg = zeros(Float64, n, n, n)  # dg[σ, μ, ν] = ∂g_{μν}/∂x^σ
    for σ in 1:n
        xp = copy(x); xp[σ] += h
        xm = copy(x); xm[σ] -= h
        dg[σ, :, :] = (g_fn(xp) .- g_fn(xm)) ./ (2h)
    end
    return dg
end

# ∂Γ^λ_{μν}/∂x^σ at point x.  Returns a 4-index array dΓ[σ,λ,μ,ν].
function _christoffel_partials(g_fn, x::Vector{Float64}, h::Float64=1e-5)
    n = length(x)
    dΓ = zeros(Float64, n, n, n, n)
    for σ in 1:n
        xp = copy(x); xp[σ] += h
        xm = copy(x); xm[σ] -= h
        Γp = _christoffel_data(g_fn, xp, h)
        Γm = _christoffel_data(g_fn, xm, h)
        dΓ[σ, :, :, :] = (Γp .- Γm) ./ (2h)
    end
    return dΓ
end

# Γ^λ_{μν} = (1/2) g^{λσ} (∂_μ g_{σν} + ∂_ν g_{σμ} - ∂_σ g_{μν})
function _christoffel_data(g_fn, x::Vector{Float64}, h::Float64=1e-5)
    n    = length(x)
    g_mat = g_fn(x)
    g_inv = inv(g_mat)
    dg    = _metric_partials(g_fn, x, h)  # dg[σ, μ, ν] = ∂_σ g_{μν}
    Γ = zeros(Float64, n, n, n)
    for λ in 1:n, μ in 1:n, ν in 1:n
        s = 0.0
        for σ in 1:n
            # ∂_μ g_{σν}  →  dg[μ, σ, ν]
            # ∂_ν g_{σμ}  →  dg[ν, σ, μ]
            # ∂_σ g_{μν}  →  dg[σ, μ, ν]
            s += g_inv[λ, σ] * (dg[μ, σ, ν] + dg[ν, σ, μ] - dg[σ, μ, ν])
        end
        Γ[λ, μ, ν] = 0.5 * s
    end
    return Γ
end

# ── christoffel_symbols ───────────────────────────────────────────────────────
function christoffel_symbols(g_fn, x::Vector{Float64}; h::Float64=1e-5)
    Γ_data = _christoffel_data(g_fn, x, h)
    indices = [upper(:λ), lower(:μ), lower(:ν)]
    return Tensor(Γ_data, indices, :Γ)
end

# ── riemann_tensor ────────────────────────────────────────────────────────────
#    R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}
function riemann_tensor(g_fn, x::Vector{Float64}; h::Float64=1e-5)
    n   = length(x)
    Γ   = _christoffel_data(g_fn, x, h)       # Γ[λ, μ, ν]
    dΓ  = _christoffel_partials(g_fn, x, h)   # dΓ[σ, λ, μ, ν]

    R = zeros(Float64, n, n, n, n)   # R[ρ, σ, μ, ν]
    for ρ in 1:n, σ in 1:n, μ in 1:n, ν in 1:n
        # Term 1:  ∂_μ Γ^ρ_{νσ}  →  dΓ[μ, ρ, ν, σ]
        # Term 2:  ∂_ν Γ^ρ_{μσ}  →  dΓ[ν, ρ, μ, σ]
        term1 =  dΓ[μ, ρ, ν, σ]
        term2 = -dΓ[ν, ρ, μ, σ]
        # Term 3:  Γ^ρ_{μλ} Γ^λ_{νσ}
        # Term 4: -Γ^ρ_{νλ} Γ^λ_{μσ}
        term3 = 0.0
        term4 = 0.0
        for λ in 1:n
            term3 += Γ[ρ, μ, λ] * Γ[λ, ν, σ]
            term4 -= Γ[ρ, ν, λ] * Γ[λ, μ, σ]
        end
        R[ρ, σ, μ, ν] = term1 + term2 + term3 + term4
    end
    indices = [upper(:ρ), lower(:σ), lower(:μ), lower(:ν)]
    return Tensor(R, indices, :R)
end

# ── ricci_tensor ──────────────────────────────────────────────────────────────
# Compute R_{μν} = R^ρ_{μρν} (contraction of Riemann on indices 1 and 3).
function ricci_tensor(g_fn, x::Vector{Float64}; h::Float64=1e-5)
    Riem = riemann_tensor(g_fn, x; h=h)
    n = size(Riem.data, 1)

    # R_{μν} = R^ρ_{μρν}: sum over ρ, keeping free indices σ=μ and ν.
    # In our array layout:  R_data[ρ, σ, μ, ν]
    # Ricci[μ, ν] = Σ_ρ  R_data[ρ, μ, ρ, ν]
    Ric = zeros(Float64, n, n)
    @inbounds for μ in 1:n, ν in 1:n
        s = 0.0
        for ρ in 1:n
            s += Riem.data[ρ, μ, ρ, ν]
        end
        Ric[μ, ν] = s
    end
    indices = [lower(:μ), lower(:ν)]
    return Tensor(Ric, indices, :Ric)
end

# ── ricci_scalar ──────────────────────────────────────────────────────────────
# R = g^{μν} R_{μν}.
function ricci_scalar(g_fn, x::Vector{Float64}; h::Float64=1e-5)
    g_mat  = g_fn(x)
    g_inv  = inv(g_mat)
    Ric    = ricci_tensor(g_fn, x; h=h)
    R = 0.0
    n = size(g_mat, 1)
    @inbounds for μ in 1:n, ν in 1:n
        R += g_inv[μ, ν] * Ric.data[μ, ν]
    end
    return R
end

# ── einstein_tensor ───────────────────────────────────────────────────────────
# G_{μν} = R_{μν} - (1/2) g_{μν} R.
function einstein_tensor(g_fn, x::Vector{Float64}; h::Float64=1e-5)
    g_mat = g_fn(x)
    Ric = ricci_tensor(g_fn, x; h=h)
    R = ricci_scalar(g_fn, x; h=h)
    G_data = Ric.data .- 0.5 .* g_mat .* R
    indices = [lower(:μ), lower(:ν)]
    return Tensor(G_data, indices, :G)
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    include("TensorOps.jl")
    using .TensorOps
    using .StandardTensors
    println("Demo 1 — Minkowski metric (flat space)")
    println("All Christoffel symbols and curvature should be zero.")
    η = [-1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]
    g_mink = x -> η

    x0 = [0.0, 0.0, 0.0, 0.0]
    Γ   = christoffel_symbols(g_mink, x0)
    println("\nChristoffel Γ^λ_{μν} (all should be 0):")
    println("  max |Γ| = ", maximum(abs, Γ.data))

    R_mink = riemann_tensor(g_mink, x0)
    println("Riemann R^ρ_{σμν} (all should be 0):")
    println("  max |R| = ", maximum(abs, R_mink.data))


    println()
    println("Demo 2 — Schwarzschild metric (exterior, M=1, G=c=1)")
    println("Coordinate order: (t, r, θ, φ)")

    M = 1.0
    function g_schwarz(x::Vector{Float64})
        _, r, θ, _ = x
        f  = 1.0 - 2M / r
        return [-f    0.0        0.0          0.0;
                 0.0  1.0/f      0.0          0.0;
                 0.0  0.0        r^2          0.0;
                 0.0  0.0        0.0          r^2 * sin(θ)^2]
    end
    x_s = [0.0, 6.0, π/2, 0.0]    # r = 6M, equatorial plane
    Γ_s  = christoffel_symbols(g_schwarz, x_s)
    println("\nNon-zero Christoffel symbols at r=6M, θ=π/2:")
    n = 4
    coord_names = ["t", "r", "θ", "φ"]
    for λ in 1:n, μ in 1:n, ν in 1:n
        v = Γ_s.data[λ, μ, ν]
        if abs(v) > 1e-10
            println("  Γ^$(coord_names[λ])_{$(coord_names[μ])$(coord_names[ν])} = $v")
        end
    end

    Ric_s = ricci_tensor(g_schwarz, x_s)
    println("\nRicci tensor R_{μν} (should vanish for vacuum Schwarzschild):")
    println("  max |Ric| = ", maximum(abs, Ric_s.data))

    Rs = ricci_scalar(g_schwarz, x_s)
    println("Ricci scalar R (should be 0): $Rs")

    G_s = einstein_tensor(g_schwarz, x_s)
    println("Einstein tensor G_{μν} (should be 0 in vacuum):")
    println("  max |G| = ", maximum(abs, G_s.data))


    println()
    println("Demo 3 — FLRW metric (flat, k=0, a(t)=e^{Ht}, de Sitter)")
    println("Coordinate order: (t, x, y, z)")

    H = 1.0
    function g_flrw(x::Vector{Float64})
        t, _, _, _ = x
        a2 = exp(2H * t)
        return [-1.0  0.0   0.0   0.0;
                 0.0  a2    0.0   0.0;
                 0.0  0.0   a2    0.0;
                 0.0  0.0   0.0   a2]
    end

    x_f = [0.0, 0.0, 0.0, 0.0]

    Γ_f = christoffel_symbols(g_flrw, x_f)
    println("\nNon-zero Christoffel symbols at t=0 (de Sitter, H=1):")
    for λ in 1:n, μ in 1:n, ν in 1:n
        v = Γ_f.data[λ, μ, ν]
        if abs(v) > 1e-10
            println("  Γ^$(coord_names[λ])_{$(coord_names[μ])$(coord_names[ν])} = $(round(v; digits=6))")
        end
    end

    Ric_f  = ricci_tensor(g_flrw, x_f)
    Rs_f   = ricci_scalar(g_flrw, x_f)
    G_f    = einstein_tensor(g_flrw, x_f)
    println("\nRicci scalar R (de Sitter, H=1): $(round(Rs_f; digits=6))")
    println("Expected ≈ 12H² = $(12H^2)")
    println("Einstein tensor G_{μν} (diagonal, spatial components = 3H²):")
    for μ in 1:n
        v = G_f.data[μ, μ]
        println("  G_{$(coord_names[μ])$(coord_names[μ])} = $(round(v; digits=6))")
    end
end