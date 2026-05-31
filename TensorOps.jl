module TensorOps

export TensorKind, TensorIndex, Tensor, lower, upper, self_contract, contract, tensor_assign, to_latex 

@enum TensorKind begin
    TrueTensor
    Symbol_3
    Density
end

struct TensorIndex
    name::Symbol
    is_contravariant::Bool
end

struct Tensor{T<:AbstractArray}
    data::T
    indices::Vector{TensorIndex}
    name::Symbol
    kind::TensorKind
    weight::Rational{Int}
end
Tensor(data, indices, name) = Tensor(data, indices, name, TrueTensor, 0//1)

function Base.show(io::IO, t::Tensor)
    upper_indices = [idx.name for idx in t.indices if idx.is_contravariant]
    lower_indices = [idx.name for idx in t.indices if !idx.is_contravariant]
    index_str = ""
    if !isempty(upper_indices)
        index_str *= "^{" * join(string.(upper_indices)) * "}"
    end
    if !isempty(lower_indices)
        index_str *= "_{" * join(string.(lower_indices)) * "}"
    end
    print(io, "Tensor $(t.name)$index_str:\n")
    print(io, t.data)
end

function validate_indices(indices::Vector{TensorIndex})
    counts = Dict{Symbol, Vector{Bool}}()
    for idx in indices
        push!(get!(counts, idx.name, Bool[]), idx.is_contravariant)
    end
    free_indices  = Symbol[]
    dummy_indices = Tuple{Symbol, Bool, Bool}[]
    for (name, positions) in counts
        n = length(positions)
        if n == 1
            push!(free_indices, name)
        elseif n == 2
            up, down = positions[1], positions[2]
            if up == down
                @warn "Dummy index $name appears twice as $(up ? "upper" : "lower") -- not a proper Einstein contraction"
            end
            push!(dummy_indices, (name, positions[1], positions[2]))
        else
            error("Index $name appears $n times -- Einstein convention allows max 2")
        end
    end
    return free_indices, dummy_indices
end

function _contract_loops!(
    C::AbstractArray,
    A::AbstractArray,
    B::AbstractArray,
    A_labels::Vector{Symbol},
    B_labels::Vector{Symbol},
    out_labels::Vector{Symbol},
    dummy_labels::Vector{Symbol},
    dim_map::Dict{Symbol, Int}
)
    ranges      = Dict(l => 1:dim_map[l] for l in keys(dim_map))
    idx(lbls,d) = Tuple(d[l] for l in lbls)
    free_ranges  = Tuple(ranges[l] for l in out_labels)
    dummy_ranges = Tuple(ranges[l] for l in dummy_labels)
    @inbounds for free_ci in CartesianIndices(free_ranges)
        free_dict = Dict(out_labels[i] => free_ci[i] for i in eachindex(out_labels))
        acc = zero(eltype(C))
        for dummy_ci in CartesianIndices(dummy_ranges)
            dummy_dict = Dict(dummy_labels[i] => dummy_ci[i] for i in eachindex(dummy_labels))
            all_idx = merge(free_dict, dummy_dict)
            acc += A[idx(A_labels, all_idx)...] * B[idx(B_labels, all_idx)...]
        end
        C[free_ci] = acc
    end
end

# `all_labels`   – the expression-level label for every axis of A (length = ndims(A))
# `out_labels`   – the subset of labels that survive (free indices)
# `dummy_labels` – the labels that are summed over (appear twice in all_labels)
function _self_contract_loops!(
    C::AbstractArray,
    A::AbstractArray,
    all_labels::Vector{Symbol},
    out_labels::Vector{Symbol},
    dummy_labels::Vector{Symbol},
    dim_map::Dict{Symbol, Int}
)
    ranges       = Dict(l => 1:dim_map[l] for l in keys(dim_map))
    idx(lbls, d) = Tuple(d[l] for l in lbls)
    free_ranges  = isempty(out_labels)   ? () : Tuple(ranges[l] for l in out_labels)
    dummy_ranges = isempty(dummy_labels) ? () : Tuple(ranges[l] for l in dummy_labels)
    if isempty(out_labels)
        # Result is a scalar stored as a 0-d array.
        acc = zero(eltype(C))
        for dummy_ci in CartesianIndices(dummy_ranges)
            d = Dict(dummy_labels[i] => dummy_ci[i] for i in eachindex(dummy_labels))
            acc += A[idx(all_labels, d)...]
        end
        C[] = acc
    else
        @inbounds for free_ci in CartesianIndices(free_ranges)
            free_dict = Dict(out_labels[i] => free_ci[i] for i in eachindex(out_labels))
            acc = zero(eltype(C))
            for dummy_ci in CartesianIndices(dummy_ranges)
                dummy_dict = Dict(dummy_labels[i] => dummy_ci[i] for i in eachindex(dummy_labels))
                all_idx = merge(free_dict, dummy_dict)
                acc += A[idx(all_labels, all_idx)...]
            end
            C[free_ci] = acc
        end
    end
end

# This is used when an expression supplies different label names than the ones
# the tensor was constructed with (e.g. the tensor stores μ,ν but the
# expression says α,β).
function rename_indices(t::Tensor, new_indices::Vector{TensorIndex})
    if length(new_indices) != length(t.indices)
        error("rename_indices: $(t.name) has $(length(t.indices)) indices but $(length(new_indices)) were supplied")
    end
    return Tensor(t.data, new_indices, t.name, t.kind, t.weight)
end

function self_contract(A::Tensor, expr_indices::Vector{TensorIndex})
    if length(expr_indices) != ndims(A.data)
        error("self_contract: $(A.name) has $(ndims(A.data)) dimensions but $(length(expr_indices)) indices were given")
    end
    _, dummy_pairs = validate_indices(expr_indices)
    dummy_names = Set([d[1] for d in dummy_pairs])
    all_labels = [idx.name for idx in expr_indices]
    out_labels = unique([idx.name for idx in expr_indices if idx.name ∉ dummy_names])

    label_contravariance = Dict{Symbol, Bool}()
    for idx in expr_indices
        if idx.name ∉ dummy_names
            label_contravariance[idx.name] = idx.is_contravariant
        end
    end

    dim_map = Dict{Symbol, Int}()
    for (i, label) in enumerate(all_labels)
        dim_map[label] = size(A.data, i)
    end

    if isempty(out_labels)
        C = zeros(eltype(A.data))   # scalar result
    else
        out_shape = Tuple(dim_map[l] for l in out_labels)
        C = zeros(eltype(A.data), out_shape...)
    end
    _self_contract_loops!(C, A.data, all_labels, out_labels, collect(dummy_names), dim_map)
    out_indices = [TensorIndex(l, label_contravariance[l]) for l in out_labels]
    return Tensor(C, out_indices, :result)
end

function contract(A::Tensor, A_expr_indices::Vector{TensorIndex},
                  B::Tensor, B_expr_indices::Vector{TensorIndex})
    all_indices = [A_expr_indices; B_expr_indices]
    free_idxs, dummy_idxs = validate_indices(all_indices)
    A_labels    = [idx.name for idx in A_expr_indices]
    B_labels    = [idx.name for idx in B_expr_indices]
    dummy_names = Set([d[1] for d in dummy_idxs])
    out_labels  = unique([l for l in [A_labels; B_labels] if l ∉ dummy_names])

    label_contravariance = Dict{Symbol, Bool}()
    for idx in A_expr_indices; label_contravariance[idx.name] = idx.is_contravariant; end
    for idx in B_expr_indices; label_contravariance[idx.name] = idx.is_contravariant; end

    dim_map = Dict{Symbol, Int}()
    for (i, label) in enumerate(A_labels)
        dim_map[label] = size(A.data, i)
    end
    for (i, label) in enumerate(B_labels)
        if haskey(dim_map, label)
            if dim_map[label] != size(B.data, i)
                error("Dimension mismatch for label $label: $(dim_map[label]) vs $(size(B.data, i))")
            end
        else
            dim_map[label] = size(B.data, i)
        end
    end

    out_shape = Tuple(dim_map[l] for l in out_labels)
    C = zeros(eltype(A.data), out_shape...)
    _contract_loops!(C, A.data, B.data, A_labels, B_labels, out_labels, collect(dummy_names), dim_map)
    out_indices = [TensorIndex(l, label_contravariance[l]) for l in out_labels]
    return Tensor(C, out_indices, :result)
end

lower(name::Symbol) = TensorIndex(name, false)
upper(name::Symbol) = TensorIndex(name, true)

# Supports plain letters (μ, ν, α …) and primed labels written as μ' or μp.
# A label is one Unicode letter optionally followed by ' or p (apostrophe/prime).
function parse_tensor_indices(index_str::Union{String, SubString})
    indices = TensorIndex[]
    is_contra = false
    i = firstindex(index_str)
    while i <= lastindex(index_str)
        c = index_str[i]
        if c == '^'
            is_contra = true
            i = nextind(index_str, i)
        elseif c == '_'
            is_contra = false
            i = nextind(index_str, i)
        elseif c == '{' || c == '}' || isspace(c)
            i = nextind(index_str, i)
        else
            # Read one base character.
            base = string(c)
            i = nextind(index_str, i)
            # Check for a prime marker: ' or the letter p immediately after.
            if i <= lastindex(index_str)
                nc = index_str[i]
                if nc == '\'' || nc == 'p'
                    base *= string(nc)
                    i = nextind(index_str, i)
                end
            end
            push!(indices, TensorIndex(Symbol(base), is_contra))
        end
    end
    return indices
end

function parse_tensor_term(term::Union{String, SubString}, namespace::Dict)
    term = strip(String(term))
    if contains(term, '{')
        idx_start = findfirst('{', term)
        idx_end   = findlast('}', term)
        name_str  = term[1:prevind(term, idx_start)]
        index_str = term[nextind(term, idx_start):prevind(term, idx_end)]
    else
        name_str  = string(first(term))
        index_str = term[nextind(term, 1):end]
    end
    name = Symbol(replace(name_str, "^" => "", "_" => ""))
    if !haskey(namespace, name)
        error("Tensor '$name' not found in namespace")
    end
    original_tensor = namespace[name]
    parsed_indices = parse_tensor_indices(index_str)
    if length(parsed_indices) != length(original_tensor.indices)
        error("Rank mismatch: Tensor '$name' expects $(length(original_tensor.indices)) indices, but $(length(parsed_indices)) were provided.")
    end
    for i in 1:length(parsed_indices)
        is_parsed_contra = parsed_indices[i].is_contravariant
        is_orig_contra = original_tensor.indices[i].is_contravariant
        if is_parsed_contra != is_orig_contra
            orig_pos = is_orig_contra ? "upper (contravariant)" : "lower (covariant)"
            parsed_pos = is_parsed_contra ? "upper" : "lower"
            @warn "Index position mismatch for tensor '$name' at slot $i. " *
                  "It was defined with a $orig_pos index, but is being used with a $parsed_pos index in the expression."
        end
    end
    return original_tensor, parsed_indices
end

# Split an expression string into space-delimited terms, respecting { } nesting.
function _split_terms(expr_str::AbstractString)
    terms = String[]
    current_term = ""
    brace_depth = 0
    for char in expr_str
        if char == '{'
            brace_depth += 1
        elseif char == '}'
            brace_depth -= 1
        elseif char == ' ' && brace_depth == 0
            if !isempty(current_term)
                push!(terms, current_term)
                current_term = ""
            end
            continue
        end
        current_term *= char
    end
    isempty(current_term) || push!(terms, current_term)
    return terms
end

function tensor_expr(expr_str::Union{String, SubString}, namespace::Dict)::Tensor
    expr_str = strip(String(expr_str))
    terms = _split_terms(expr_str)
    if length(terms) == 1
        # Single tensor: check for self-contraction (repeated index).
        A, A_indices = parse_tensor_term(terms[1], namespace)
        _, dummy_pairs = validate_indices(A_indices)
        if isempty(dummy_pairs)
            # No contraction at all; just return the tensor renamed with the expression indices
            return rename_indices(A, A_indices)
        else
            return self_contract(A, A_indices)
        end
    end

    # Two or more tensors: chain pairwise contractions.
    A, A_indices = parse_tensor_term(terms[1], namespace)
    # If the first term already has an internal self-contraction, resolve it.
    let _, dp = validate_indices(A_indices)
        if !isempty(dp)
            A = self_contract(A, A_indices)
            A_indices = A.indices
        end
    end
    result = A
    result_idxs = A_indices
    for i in 2:length(terms)
        B, B_indices = parse_tensor_term(terms[i], namespace)
        # Resolve any self-contraction within B before the pairwise step.
        let _, dp = validate_indices(B_indices)
            if !isempty(dp)
                B = self_contract(B, B_indices)
                B_indices = B.indices
            end
        end
        result = contract(result, result_idxs, B, B_indices)
        result_idxs = result.indices
    end
    return result
end

function tensor_assign(equation::Union{String, SubString}, namespace::Dict)::Tensor
    equation = String(equation)
    contains(equation, '=') || error("Equation must contain '='")
    lhs_str, rhs_str = split(equation, '='; limit=2)
    lhs_str = strip(lhs_str)
    rhs_str = strip(rhs_str)
    lhs_clean = replace(lhs_str, "^" => "", "_" => "")
    target_name = if contains(lhs_clean, '{')
        Symbol(lhs_clean[1:prevind(lhs_clean, findfirst('{', lhs_clean))])
    else
        Symbol(first(lhs_clean))
    end
    result = tensor_expr(rhs_str, namespace)
    return Tensor(result.data, result.indices, target_name, result.kind, result.weight)
end

function latex_translate(s::String)
    res = s
    if endswith(res, "'")
        res = res[1:prevind(res, lastindex(res))] * "^\\prime"
    elseif endswith(res, "p") && length(res) > 1 && !startswith(res, "\\")
        res = res[1:prevind(res, lastindex(res))] * "^\\prime"
    end

    greek = Dict(
        "α" => "\\alpha", "β" => "\\beta", "γ" => "\\gamma", "δ" => "\\delta",
        "ϵ" => "\\epsilon", "ε" => "\\varepsilon", "ζ" => "\\zeta", "η" => "\\eta",
        "θ" => "\\theta", "ι" => "\\iota", "κ" => "\\kappa", "λ" => "\\lambda",
        "μ" => "\\mu", "ν" => "\\nu", "ξ" => "\\xi", "π" => "\\pi",
        "ρ" => "\\rho", "σ" => "\\sigma", "τ" => "\\tau", "υ" => "\\upsilon",
        "ϕ" => "\\phi", "φ" => "\\varphi", "χ" => "\\chi", "ψ" => "\\psi",
        "ω" => "\\omega",
        "Γ" => "\\Gamma", "Δ" => "\\Delta", "Θ" => "\\Theta", "Λ" => "\\Lambda",
        "Ξ" => "\\Xi", "Π" => "\\Pi", "Σ" => "\\Sigma", "Υ" => "\\Upsilon",
        "Φ" => "\\Phi", "Ψ" => "\\Psi", "Ω" => "\\Omega"
    )
    for (k, v) in greek
        res = replace(res, k => v)
    end
    return res
end

function format_num(x::Number)
    rx = round(x; digits=6)
    if rx == round(rx)
        return string(Int(rx))
    else
        return string(rx)
    end
end

function data_to_latex(arr::AbstractArray)
    nd = ndims(arr)
    if nd == 0
        return format_num(arr[])
    elseif nd == 1
        return "\\begin{pmatrix} " * join([format_num(x) for x in arr], " \\\\ ") * " \\end{pmatrix}"
    elseif nd == 2
        rows = String[]
        for r in 1:size(arr, 1)
            push!(rows, join([format_num(arr[r, c]) for c in 1:size(arr, 2)], " & "))
        end
        return "\\begin{pmatrix} " * join(rows, " \\\\ ") * " \\end{pmatrix}"
    elseif nd == 3
        slices = String[]
        for i in 1:size(arr, 1)
            sub_rows = String[]
            for r in 1:size(arr, 2)
                push!(sub_rows, join([format_num(arr[i, r, c]) for c in 1:size(arr, 3)], " & "))
            end
            push!(slices, "\\text{index 1 = } $i: \\begin{pmatrix} " * join(sub_rows, " \\\\ ") * " \\end{pmatrix}")
        end
        return join(slices, "\\quad ")
    else
        return "\\text{High-rank tensor (" * string(nd) * "-dimensional). See component list below.}"
    end
end

function to_latex(t::Tensor; coords=String[])
    # Convert coords to a Vector{String} to avoid type invariance issues
    coords_str = String.(coords)
    name_str = string(t.name)
    if name_str == "g_inv" || name_str == "ig"
        name_str = "g"
    elseif name_str == "Γ" || name_str == "Gamma"
        name_str = "\\Gamma"
    elseif name_str == "Λ" || name_str == "Lambda"
        name_str = "\\Lambda"
    elseif name_str == "Ric"
        name_str = "R"
    end

    # FIX 1: Wrap the base name in curly braces. 
    # This prevents double subscript errors if name_str already contains an underscore.
    name_str = "{" * name_str * "}"

    # Collect contravariant and covariant indices separately
    contravariant_indices = String[]
    covariant_indices = String[]
    for idx in t.indices
        # FIX 2: Wrap individual indices in braces to prevent macro bleeding
        latex_idx = "{" * latex_translate(string(idx.name)) * "}"
        if idx.is_contravariant
            push!(contravariant_indices, latex_idx)
        else
            push!(covariant_indices, latex_idx)
        end
    end
    
    # Build index string with proper LaTeX notation
    idx_str = ""
    if !isempty(contravariant_indices)
        idx_str *= "^{" * join(contravariant_indices, "") * "}"
    end
    if !isempty(covariant_indices)
        idx_str *= "_{" * join(covariant_indices, "") * "}"
    end

    full_lhs = name_str * idx_str
    data_latex = data_to_latex(t.data)

    components = String[]
    if ndims(t.data) > 0
        for ci in CartesianIndices(t.data)
            val = t.data[ci]
            if abs(val) > 1e-9
                comp_contravariant = String[]
                comp_covariant = String[]
                for (dim_idx, idx) in enumerate(t.indices)
                    c_name = (!isempty(coords_str) && dim_idx <= length(coords_str)) ? coords_str[ci[dim_idx]] : string(ci[dim_idx])
                    
                    # FIX 3: Apply the same brace-wrapping to component indices
                    latex_c_name = "{" * latex_translate(c_name) * "}"
                    
                    if idx.is_contravariant
                        push!(comp_contravariant, latex_c_name)
                    else
                        push!(comp_covariant, latex_c_name)
                    end
                end
                
                comp_idx_str = ""
                if !isempty(comp_contravariant)
                    comp_idx_str *= "^{" * join(comp_contravariant, "") * "}"
                end
                if !isempty(comp_covariant)
                    comp_idx_str *= "_{" * join(comp_covariant, "") * "}"
                end
                push!(components, name_str * comp_idx_str * " = " * format_num(val))
            end
        end
    else
        push!(components, name_str * " = " * format_num(t.data[]))
    end

    return full_lhs, data_latex, components
end

end