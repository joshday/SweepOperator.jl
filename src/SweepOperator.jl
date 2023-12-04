module SweepOperator
export sweep!

using LinearAlgebra
import LinearAlgebra: BlasFloat, checksquare

const AMat = AbstractMatrix
const AVec = AbstractVector

"""
    sweep!(A, k ; inv=false)
    sweep!(A, ks; inv=false)

Perform the sweep operation (or inverse sweep if `inv=true`) on matrix `A` on element `k`
(or each element in `ks`).  Only the upper triangle is read/swept.

# Example:

    x = randn(100, 10)
    xtx = x'x
    sweep!(xtx, 1)
    sweep!(xtx, 1, true)
"""
function sweep!(A::AMat, k::Integer, inv::Bool = false)
    sweep_with_buffer!(Vector{eltype(A)}(undef, size(A, 2)), A, k, inv)
end

function sweep!(A::AMat{T}, ks::AVec{I}, inv::Bool = false) where {T<:BlasFloat, I<:Integer}
    akk = Vector{T}(undef, size(A,1))
    for k in ks
        sweep_with_buffer!(akk, A, k, inv)
    end
    A
end

function sweep_with_buffer!(akk::AVec{T}, A::AMat{T}, k::Integer, inv::Bool = false) where {T}
    # ensure @inbounds is safe
    p = checksquare(A)
    1 ≤ k ≤ p || throw(BoundsError(A, k))
    p == length(akk) || throw(DimensionError("Incorrect buffer size."))
    @inbounds @views begin
        d = one(T) / A[k, k]                # pivot
        copy!(akk, Symmetric(A, :U)[:, k])  # akk = A[:, k]
        syrk!(A, -d, akk)                   # everything not in row/col k
        rmul!(akk, d * (-one(T)) ^ inv)     # akk .* d (negated if inv=true)
        setrowcol!(A, k, akk)
        A[k, k] = -d                        # pivot element
    end
    return A
end

#-----------------------------------------------------------------------------# setrowcol!
# Set upper triangle of: (A[k, :] = x; A[:, k] = x)
function setrowcol!(A::StridedArray, k, x)
    @views copy!(A[1:k-1,k], x[1:k-1])         # col k
    @views copy!(A[k, k+1:end], x[k+1:end])    # row k
end

setrowcol!(A::Union{Hermitian,Symmetric,UpperTriangular}, k, x) = setrowcol!(A.data, k, x)

#-----------------------------------------------------------------------------# syrk!
const BlasNumber = Union{LinearAlgebra.BlasFloat, LinearAlgebra.BlasComplex}

# In-place update of (the upper triangle of) A + α * x * x'
function syrk!(A::StridedMatrix{T}, α::T, x::AbstractArray{<:T}) where {T<:BlasNumber}
    BLAS.syrk!('U', 'N', α, x, one(T), A)
end

function syrk!(A::Hermitian{T, S}, α::T, x::AbstractArray{<:T}) where {T<:BlasNumber, S<:StridedMatrix{T}}
    Hermitian(BLAS.syrk!('U', 'N', α, x, one(T), A.data))
end

function syrk!(A::Symmetric{T, S}, α::T, x::AbstractArray{<:T}) where {T<:BlasNumber, S<:StridedMatrix{T}}
    Symmetric(BLAS.syrk!('U', 'N', α, x, one(T), A.data))
end

function syrk!(A, α, x)  # fallback
    p = checksquare(A)
    for i in 1:p, j in i:p
        @inbounds A[i,j] += α * x[i] * x[j]
    end
end


end # module
