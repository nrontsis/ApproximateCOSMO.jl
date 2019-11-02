using Polynomials, LinearMaps
using LinearAlgebra, SparseArrays
mutable struct CustomBOSolver{T} <: COSMO.AbstractKKTSolver
    k::Int
    σ::T
    ρ1::T
    ρ2::T
    n::Int
    m::Int
    sqrt_dim::Int
    A1::SparseMatrixCSC{T, Int}
    A::SparseMatrixCSC{T, Int}
    P::SparseMatrixCSC{T, Int}
    tmp_n::Vector{T}
    
    function CustomBOSolver(P::AbstractMatrix{T}, A::AbstractMatrix{T}, σ::T, ρ) where T
        n = size(P, 1)
        m = size(A, 1) - n
        ρ1 = ρ[1];
        ρ2 = ρ[end];
        k = Int(round(real(roots(Poly([-n, 0, 1/2, 1/2]))[end])))
        @assert k*(k + 1)*k/2 == n
        sqrt_dim = Int(k*(k + 1)/2)
        A1 = kron(ones(1, k), SparseMatrixCSC{T, Int}(I, sqrt_dim, sqrt_dim))
        new{T}(k, σ, ρ1, ρ2, n, m, Int(k*(k + 1)/2), A1, A, P, zeros(T, n))
    end
end

function mul_A1!(y, x, S)
    y .= 0
    idx = 1
    @inbounds for i = 1:S.k, j = 1:S.sqrt_dim
        y[j] += x[idx]
        idx += 1
    end
    #@assert norm(S.A1*x - y) <= 1e-8
end

function mul_A1t!(y, x, S)
    idx = 1
    @inbounds for i = 1:S.k, j = 1:S.sqrt_dim
        y[idx] = x[j]
        idx += 1
    end
    #@assert norm(S.A1'*x - y) <= 1e-8
end

function solve!(S::CustomBOSolver, y::AbstractVector{T}, x::AbstractVector{T}) where {T}
    # Solves the (KKT) linear system
    # | P + σI     A'  |  |y1|  =  |x1|
    # | A        -I/ρ  |  |y23|  =  |x23|
    # x1,y1: R^n, x2/y2: R^m
    # where [y1; y2] := y, [x1; x2] := x 
    y1 = view(y, 1:S.n); y2 = view(y, S.n+1:S.n+S.m); y3 = view(y, S.n+S.m+1:2*S.n+S.m);
    x1 = view(x, 1:S.n); x2 = view(x, S.n+1:S.n+S.m); x3 = view(x, S.n+S.m+1:2*S.n+S.m);

    #y2 .= (S.A1*(x1 + x3*S.ρ2)/(S.ρ2 + S.σ) - x2)./(1/S.ρ1 + S.k/(S.ρ2 + S.σ))
    #y1 .= (-S.A1'*y2 + x1 + x3*S.ρ2)/(S.ρ2 + S.σ)
    #y3 .= S.ρ2* (y1 - x3)

    @. S.tmp_n = (x1 + x3*S.ρ2)/(S.ρ2 + S.σ);
    mul_A1!(y2, S.tmp_n, S)
    @. y2 = (y2 - x2)/(1/S.ρ1 + S.k/(S.ρ2 + S.σ))
    mul_A1t!(y1, y2, S)
    @. y1 = (-y1 + x1 + x3*S.ρ2)/(S.ρ2 + S.σ)
    @. y3 = S.ρ2* (y1 - x3)

    #=
    F = [S.P + S.σ*I SparseMatrixCSC(S.A'); S.A I]
    @inbounds @simd for i = (S.n + 1):size(F, 2)
        if i <= S.n + S.m
            ρ = S.ρ1    
        else
            ρ = S.ρ2
        end
        F[i, i] = -1.0 / ρ
    end
    # show(stdout, "text/plain", F); println()
    @assert norm(y - F\x) <= 1e-7
    =#
end

function update_rho!(S::CustomBOSolver, ρ)
    S.ρ1 = ρ[1]
    S.ρ2 = ρ[end]
end
