module SweepOperator
export sweep!

const AMat = AbstractMatrix
const AVec = AbstractVector


"""
# Symmetric sweep operator
Symmetric sweep operator of the matrix `A` on element `k`.  `A` is overwritten.
`inv = true` will perform the inverse sweep.  Only the upper triangle is read and swept.

`sweep!(A, k, inv = false)`

Providing a Range, rather than an Integer, sweeps on each element in the range.

`sweep!(A, first:last, inv = false)`

### Example:

```julia
x = randn(100, 10)
xtx = x'x
sweep!(xtx, 1)
sweep!(xtx, 1, true)
```
"""
function sweep!{T<:LinAlg.BlasFloat}(A::AMat{T}, k::Integer, inv::Bool = false)
    sweep_with_buffer!(zeros(T, size(A, 2)), A, k, inv)
end


function sweep_with_buffer!{T<:LinAlg.BlasFloat}(akk::AVec{T}, A::AMat{T}, k::Integer,
        inv::Bool = false)
    # ensure @inbounds is safe
    p = LinAlg.checksquare(A)
    p == length(akk) || throw(DimensionError("buffer size incorrect"))
    @inbounds d = one(T) / A[k, k]  # pivot
    # get column A[:, k] (hack because only upper triangle is available)
    for j in 1:k
        @inbounds akk[j] = A[j, k]
    end
    for j in k+1:p
        @inbounds akk[j] = A[k, j]
    end
    BLAS.syrk!('U', 'N', -d, akk, one(T), A)  # everything not in col/row k
    scale!(akk, d * (-one(T)) ^ inv)
    for i in 1:(k-1)  # col k
        @inbounds A[i, k] = akk[i]
    end
    for j in (k+1):p  # row k
        @inbounds A[k, j] = akk[j]
    end
    @inbounds A[k, k] = -d  # pivot element
    A
end

function sweep!{T<:LinAlg.BlasFloat, I<:Integer}(A::AMat{T}, ks::AVec{I}, inv::Bool = false)
    akk = zeros(T, size(A, 1))
    for k in ks
        sweep_with_buffer!(akk, A, k, inv)
    end
    A
end
function sweep_with_buffer!{T<:LinAlg.BlasFloat, I<:Integer}(akk::AVec{T}, A::AMat{T},
        ks::AVec{I}, inv::Bool = false)
    for k in ks
        sweep_with_buffer!(akk, A, k, inv)
    end
    A
end

end # module
