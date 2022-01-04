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
        d = one(T) / A[k, k]                            # pivot
        copy!(akk, Symmetric(A, :U)[:, k])            # akk = A[:, k]
        if A isa StridedMatrix{<:Union{LinearAlgebra.BlasFloat, LinearAlgebra.BlasComplex}}
            BLAS.syrk!('U', 'N', -d, akk, one(T), A)    # everything not in col/row k
        else
            A .+= UpperTriangular(-d * akk * akk')
        end
        rmul!(akk, d * (-one(T)) ^ inv)                 # akk .* d (negated if inv=true)
        copy!(A[1:k-1,k], akk[1:k-1])                 # col k
        copy!(A[k, k+1:end], akk[k+1:end])            # row k
        A[k, k] = -d                                    # pivot element
    end
    return A
end

end # module
