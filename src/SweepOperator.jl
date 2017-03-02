module SweepOperator
export sweep!

const AMat{T} = AbstractMatrix{T}
const AVec{T} = AbstractVector{T}


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
    # ensure @inbounds is safe
    p = LinAlg.checksquare(A)
    d = one(T) / A[k, k]  # pivot
    akk = Symmetric(A)[:, k]  # k-th column (use Symmetric because only triu available)
    BLAS.syrk!('U', 'N', -d, akk, one(T), A)  # everything not in col/row k
    scale!(akk, d * (-one(T)) ^ inv)
    for i in 1:k-1  # col k
        @inbounds A[i, k] = akk[i]
    end
    for j in k+1:p  # row k
        @inbounds A[k, j] = akk[j]
    end
    A[k, k] = -d  # pivot element
    A
end

function sweep!{T<:LinAlg.BlasFloat, I<:Integer}(A::AMat{T}, ks::AVec{I}, inv::Bool = false)
    for k in ks
        sweep!(A, k, inv)
    end
    A
end

end # module
