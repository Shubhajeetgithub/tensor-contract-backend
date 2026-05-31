include("TensorOps.jl")
include("StandardTensors.jl")

using .TensorOps
using .StandardTensors
using HTTP
using JSON

function get_metric(metric_type::String, params::Any)
    if metric_type == "minkowski"
        return x -> [-1.0 0.0 0.0 0.0;
                      0.0 1.0 0.0 0.0;
                      0.0 0.0 1.0 0.0;
                      0.0 0.0 0.0 1.0]
    elseif metric_type == "schwarzschild"
        M = Float64(get(params, "M", 1.0))
        return x -> begin
            _, r, θ, _ = x
            r_val = max(r, 1e-5)
            f = 1.0 - 2*M / r_val
            f_inv = isapprox(f, 0.0; atol=1e-9) ? 1e9 : 1.0 / f
            return [-f    0.0        0.0          0.0;
                     0.0  f_inv      0.0          0.0;
                     0.0  0.0        r_val^2      0.0;
                     0.0  0.0        0.0          r_val^2 * sin(θ)^2]
        end
    elseif metric_type == "flrw"
        H = Float64(get(params, "H", 1.0))
        return x -> begin
            t, _, _, _ = x
            a2 = exp(2 * H * t)
            return [-1.0  0.0   0.0   0.0;
                     0.0  a2    0.0   0.0;
                     0.0  0.0   a2    0.0;
                     0.0  0.0   0.0   a2]
        end
    elseif metric_type == "kerr"
        M = Float64(get(params, "M", 1.0))
        a = Float64(get(params, "a", 0.5))
        return x -> begin
            _, r, θ, _ = x
            r_val = max(r, 1e-5)
            sin_θ = sin(θ)
            cos_θ = cos(θ)
            Σ = r_val^2 + a^2 * cos_θ^2
            Δ = r_val^2 - 2*M*r_val + a^2
            
            g_tt = -(1.0 - 2.0*M*r_val / Σ)
            g_tϕ = -2.0*M*r_val*a*sin_θ^2 / Σ
            g_rr = Σ / (isapprox(Δ, 0.0; atol=1e-9) ? 1e9 : Δ)
            g_θθ = Σ
            g_ϕϕ = (r_val^2 + a^2 + 2.0*M*r_val*a^2*sin_θ^2 / Σ) * sin_θ^2
            
            g_mat = zeros(Float64, 4, 4)
            g_mat[1, 1] = g_tt
            g_mat[1, 4] = g_tϕ
            g_mat[4, 1] = g_tϕ
            g_mat[2, 2] = g_rr
            g_mat[3, 3] = g_θθ
            g_mat[4, 4] = g_ϕϕ
            return g_mat
        end
    elseif metric_type == "custom"
        tt_expr = get(params, "tt", "-1.0")
        rr_expr = get(params, "rr", "1.0")
        θθ_expr = get(params, "θθ", "r^2")
        ϕϕ_expr = get(params, "ϕϕ", "r^2 * sin(θ)^2")
        
        return x -> begin
            t, r, θ, ϕ = x[1], x[2], x[3], x[4]
            th = θ
            ph = ϕ
            
            eval_comp(s::String) = begin
                try
                    mod = Module()
                    Core.eval(mod, :(t = $t; r = $r; θ = $θ; ϕ = $ϕ; th = $θ; ph = $ϕ; sin = Base.sin; cos = Base.cos; tan = Base.tan; exp = Base.exp; log = Base.log; pi = Base.pi; M = $(get(params, "M", 1.0)); H = $(get(params, "H", 1.0)); a = $(get(params, "a", 0.5))))
                    return Float64(Core.eval(mod, Meta.parse(s)))
                catch e
                    @warn "Error evaluating component expression '$s': $e"
                    return 0.0
                end
            end
            
            g_mat = zeros(Float64, 4, 4)
            g_mat[1, 1] = eval_comp(tt_expr)
            g_mat[2, 2] = eval_comp(rr_expr)
            g_mat[3, 3] = eval_comp(θθ_expr)
            g_mat[4, 4] = eval_comp(ϕϕ_expr)
            return g_mat
        end
    else
        error("Unknown metric type: $metric_type")
    end
end

function handle_compute(req::HTTP.Request)
    try
        req_data = JSON.parse(String(req.body))
        metric_type = get(req_data, "metric", "minkowski")
        params = get(req_data, "params", Dict())
        coords = Float64.(get(req_data, "coords", [0.0, 0.0, 0.0, 0.0]))
        coord_names = String.(get(req_data, "coord_names", ["t", "r", "\\theta", "\\phi"]))
        expr = get(req_data, "expression", "")
        custom_vectors = get(req_data, "vectors", Dict())

        # Construct metric
        g_fn = get_metric(metric_type, params)
        g_data = g_fn(coords)
        
        # Build baseline tensors
        g_tensor = Tensor(g_data, [lower(:μ), lower(:ν)], :g)
        ig_tensor = metric_inverse(g_data)
        ig_tensor = Tensor(ig_tensor.data, [upper(:μ), upper(:ν)], :ig)

        Γ_tensor = christoffel_symbols(g_fn, coords)
        R_tensor = riemann_tensor(g_fn, coords)
        Ric_tensor = ricci_tensor(g_fn, coords)
        G_tensor = einstein_tensor(g_fn, coords)

        # Build namespace
        ns = Dict{Symbol, Any}(
            :g => g_tensor,
            :ig => ig_tensor,
            :Γ => Γ_tensor,
            :Gamma => Γ_tensor,
            :R => R_tensor,
            :Ric => Ric_tensor,
            :G => G_tensor
        )

        # Add custom tensors (like vectors)
        for (name_str, val_arr) in custom_vectors
            name_sym = Symbol(name_str)
            if val_arr isa AbstractDict
                # Detailed custom tensor/vector
                type_str = get(val_arr, "type", "vector")
                indices_raw = get(val_arr, "indices", [])
                data_raw = get(val_arr, "data", [])
                
                idx_symbols = [:μ, :ν, :ρ, :σ]
                indices_list = TensorIndex[]
                for (i, is_contra) in enumerate(indices_raw)
                    sym = i <= length(idx_symbols) ? idx_symbols[i] : Symbol("index_$i")
                    is_c = (is_contra === true || is_contra == "upper" || is_contra == "contravariant")
                    push!(indices_list, TensorIndex(sym, is_c))
                end
                
                # General Rank N tensor support!
                rank = length(indices_list)
                if rank < 1
                    rank = 1
                end
                
                flat_data = Float64.(collect(data_raw))
                expected_len = 4^rank
                
                if length(flat_data) != expected_len
                    # Safe padding or truncating if user inputs are incomplete
                    actual_len = length(flat_data)
                    if actual_len < expected_len
                        append!(flat_data, zeros(Float64, expected_len - actual_len))
                    else
                        flat_data = flat_data[1:expected_len]
                    end
                end
                
                if rank == 1
                    ns[name_sym] = Tensor(flat_data, indices_list, name_sym)
                else
                    # Reshape row-major flat data to N-dimensional Julia array.
                    # Reshape to reversed dimensions first, then permute back to reverse.
                    # This maps row-major flat data directly to column-major Julia layout.
                    rev_shape = Tuple(fill(4, rank))
                    temp_arr = reshape(flat_data, rev_shape...)
                    arr_data = permutedims(temp_arr, Tuple(reverse(1:rank)))
                    ns[name_sym] = Tensor(arr_data, indices_list, name_sym)
                end
            else
                # Fallback to simple array format
                arr_data = Float64.(val_arr)
                if ndims(arr_data) == 1
                    ns[name_sym] = Tensor(arr_data, [upper(:μ)], name_sym)
                elseif ndims(arr_data) == 2
                    ns[name_sym] = Tensor(arr_data, [upper(:μ), upper(:ν)], name_sym)
                end
            end
        end

        # Calculate expression
        local result_tensor
        if contains(expr, '=')
            result_tensor = tensor_assign(expr, ns)
        else
            result_tensor = tensor_expr(expr, ns)
        end

        # LaTeX formats
        res_lhs, res_rhs, res_comps = to_latex(result_tensor; coords=coord_names)

        # Base workspace tensors
        workspace_tensors = Dict{String, Any}()
        for (k, t) in ns
            name_str = string(k)
            if name_str == "Gamma"
                continue
            end
            lhs, rhs, comps = to_latex(t; coords=coord_names)
            workspace_tensors[name_str] = Dict(
                "lhs" => lhs,
                "rhs" => rhs,
                "components" => comps,
                "data" => t.data,
                "indices" => [Dict("name" => string(idx.name), "is_contravariant" => idx.is_contravariant) for idx in t.indices]
            )
        end

        response = Dict(
            "success" => true,
            "latex_lhs" => res_lhs,
            "latex_rhs" => res_rhs,
            "components_latex" => res_comps,
            "name" => string(result_tensor.name),
            "data" => result_tensor.data,
            "indices" => [Dict("name" => string(idx.name), "is_contravariant" => idx.is_contravariant) for idx in result_tensor.indices],
            "workspace_tensors" => workspace_tensors
        )

        return HTTP.Response(200, 
            ["Content-Type" => "application/json", "Access-Control-Allow-Origin" => "*"], 
            JSON.json(response)
        )
    catch e
        @warn "Computation failed: $e"
        response = Dict(
            "success" => false,
            "error" => string(e)
        )
        return HTTP.Response(400, 
            ["Content-Type" => "application/json", "Access-Control-Allow-Origin" => "*"], 
            JSON.json(response)
        )
    end
end

function handle_cors(req::HTTP.Request)
    return HTTP.Response(200, [
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type",
        "Access-Control-Max-Age" => "86400"
    ])
end

function main()
    port = parse(Int, get(ENV, "PORT", "8080"))
    host = "0.0.0.0"
    
    router = HTTP.Router()
    HTTP.register!(router, "GET", "/api/health", req -> HTTP.Response(200, ["Content-Type" => "application/json", "Access-Control-Allow-Origin" => "*"], "{\"status\":\"ok\"}"))
    HTTP.register!(router, "OPTIONS", "/api/compute", handle_cors)
    HTTP.register!(router, "POST", "/api/compute", handle_compute)

    println("Starting Julia GR Server on $host:$port...")
    HTTP.serve(router, host, port)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
