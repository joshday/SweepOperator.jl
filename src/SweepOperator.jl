module SweepOperator
export sweep!

using LinearAlgebra
import LinearAlgebra: BlasFloat, checksquare

const AMat = AbstractMatrix
const AVec = AbstractVector


"""
# Symmetric sweep operator
Symmetric sweep operator of the matrix `A` on element `k`.  `A` is overwritten.
`inv = true` will perform the inverse sweep.  Only the upper triangle is read and swept.

`sweep!(A, k, inv = false)`

Providing a `Range`, rather than an `Integer`, sweeps on each element in the range.

`sweep!(A, first:last, inv = false)`

### Example:

```julia
x = randn(100, 10)
xtx = x'x
sweep!(xtx, 1)
sweep!(xtx, 1, true)
```
"""
function sweep!(A::AMat, k::Integer, inv::Bool = false)
    sweep_with_buffer!(Vector{eltype(A)}(undef, size(A, 2)), A, k, inv)
end


function sweep_with_buffer!(akk::AVec{T}, A::AMat{T}, k::Integer, inv::Bool = false) where 
        {T<:BlasFloat}
    # ensure @inbounds is safe
    p = checksquare(A)
    p == length(akk) || throw(DimensionError("incorrect buffer size"))
    @inbounds d = one(T) / A[k, k]  # pivot
    # get column A[:, k] (hack because only upper triangle is available)
    for j in 1:k
        @inbounds akk[j] = A[j, k]
    end
    for j in (k+1):p
        @inbounds akk[j] = A[k, j]
    end
    BLAS.syrk!('U', 'N', -d, akk, one(T), A)  # everything not in col/row k
    rmul!(akk, d * (-one(T)) ^ inv)
    for i in 1:(k-1)  # col k
        @inbounds A[i, k] = akk[i]
    end
    for j in (k+1):p  # row k
        @inbounds A[k, j] = akk[j]
    end
    @inbounds A[k, k] = -d  # pivot element
    A
end

function sweep!(A::AMat{T}, ks::AVec{I}, inv::Bool = false) where {T<:BlasFloat, I<:Integer}
    akk = zeros(T, size(A, 1))
    for k in ks
        sweep_with_buffer!(akk, A, k, inv)
    end
    A
end
function sweep_with_buffer!(akk::AVec{T}, A::AMat{T}, ks::AVec{I}, inv::Bool = false) where 
        {T<:BlasFloat, I<:Integer}
    for k in ks
        sweep_with_buffer!(akk, A, k, inv)
    end
    A
end

end # module
