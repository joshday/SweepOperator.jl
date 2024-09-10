using Test, LinearAlgebra, Random, BenchmarkTools, SweepOperator

versioninfo()


#-----------------------------------------------------------------------------# 1) Matrix Inverse
# Code from https://github.com/joshday/SweepOperator.jl/issues/9


"""
    inv_by_chol!(A)

Invert a pd matrix `A` in-place by Cholesky decomposition.
"""
function inv_by_chol!(A::Matrix{T}) where T <: LinearAlgebra.BlasReal
    LAPACK.potrf!('U', A)
    LAPACK.potri!('U', A)
    A
end

"""
    sweep_block_kernel!(A, krange, invsw)

Perform the block form of sweep on a contiguous range of indices `krange`.

[ Ak̲k̲ Ak̲k Ak̲k̅ ]          [ Ak̲k̲-Ak̲k*Akk⁻¹*Akk̲  Ak̲k*Akk⁻¹ Ak̲k̅-Ak̲k*Akk⁻¹*Akk̅ ]
[  -  Akk Akk̅ ] -sweep-> [        -            -Akk⁻¹       Akk⁻¹*Akk̅     ]
[  -   -  Ak̅k̅ ]          [        -              -      Ak̅k̅-Ak̅k*Akk⁻¹*Akk̅ ]
"""
function sweep_block_kernel!(
        A      :: AbstractMatrix{T},
        krange :: AbstractUnitRange{<:Integer},
        invsw  :: Bool = false
        ) where {T <: LinearAlgebra.BlasReal}
    k̲range = 1:(krange[1] - 1)
    k̅range = (krange[end]+1):size(A, 2)
    Akk = view(A, krange, krange)
    Ak̲k = view(A, k̲range, krange)
    Ak̲k̲ = view(A, k̲range, k̲range)
    Akk̅ = view(A, krange, k̅range)
    Ak̅k̅ = view(A, k̅range, k̅range)
    Ak̲k̅ = view(A, k̲range, k̅range)
    # U = cholesky(Akk).U
    U, _ = LAPACK.potrf!('U', Akk)
    # Ak̲k = Ak̲k * U⁻¹
    BLAS.trsm!('R', 'U', 'N', 'N', one(T), U, Ak̲k)
    # Ak̲k̲ = Ak̲k̲ - Ak̲k * U⁻¹ * U⁻ᵀ * Akk̲
    BLAS.syrk!('U', 'N', -one(T), Ak̲k, one(T), Ak̲k̲)
    # Akk̅ = U⁻ᵀ * Akk̅
    BLAS.trsm!('L', 'U', 'T', 'N', one(T), U, Akk̅)
    # Ak̲k̅ = Ak̲k̅ - Ak̲k * U⁻¹ * U⁻ᵀ * Akk̅
    BLAS.gemm!('N', 'N', -one(T), Ak̲k, Akk̅, one(T), Ak̲k̅)
    # Ak̅k̅ = Ak̅k̅ - Ak̅k * U⁻¹ * U⁻ᵀ * Akk̅
    BLAS.syrk!('U', 'T', -one(T), Akk̅, one(T), Ak̅k̅)
    # Ak̲k = Ak̲k * Akk⁻¹ = Ak̲k * U⁻¹ * U⁻ᵀ
    s = ifelse(invsw, -one(T), one(T))
    BLAS.trsm!('R', 'U', 'T', 'N', s, Akk, Ak̲k)
    # Akk̅ = Akk⁻¹ * Akk̅ = U⁻¹ * U⁻ᵀ * Akk̅
    BLAS.trsm!('L', 'U', 'N', 'N', s, Akk, Akk̅)
    # Akk = Akk⁻¹ = U⁻ᵀ U⁻¹
    LAPACK.potri!('U', U)
    UpperTriangular(Akk) .*= -1
    A
end

# create an nxn pos-def test matrix
function run_benchmark(n::Int, seed::Int = 123)
    Random.seed!(seed)
    A = randn(n, n)
    A = A'A + I
    Ainv = UpperTriangular(inv(A))
    @test UpperTriangular(inv_by_chol!(copy(A)))        ≈ Ainv
    @test -UpperTriangular(sweep!(copy(A), 1:n))        ≈ Ainv
    @test -UpperTriangular(sweep_block!(copy(A), 1:n))  ≈ Ainv

    out = Dict()

    out["Cholesky"]     = @benchmark inv_by_chol!(copy(A))
    out["Sweep"]        = @benchmark sweep!(copy(A), 1:n)
    out["Block Sweep"]  = @benchmark sweep_block!(copy(A), 1:n)

    return out
end
