# SweepOperator

| Build | Test |
|-------|------|
| [![Build Status](https://travis-ci.org/joshday/SweepOperator.jl.svg?branch=master)](https://travis-ci.org/joshday/SweepOperator.jl) [![Build status](https://ci.appveyor.com/api/projects/status/at5bcso64joc6wbj/branch/master?svg=true)](https://ci.appveyor.com/project/joshday/sweepoperator-jl/branch/master) | [![codecov.io](http://codecov.io/github/joshday/SweepOperator.jl/coverage.svg?branch=master)](http://codecov.io/github/joshday/SweepOperator.jl?branch=master) |


The symmetric sweep operator is a powerful tool in computational statistics with uses in stepwise regression, conditional multivariate normal distributions, MANOVA, and more.  This package exports a single function:

## `sweep!`

```julia
sweep!(A, k ; inv=false)
sweep!(A, ks; inv=false)
```

Perform the sweep operation (or inverse sweep if `inv=true`) on symmetric matrix `A` on
element `k` (or each element in `ks`).  **Only the upper triangle is read/swept.**

### Example:

```julia
x = randn(100, 10)
xtx = x'x
sweep!(xtx, 1)
sweep!(xtx, 1, true)
```


# Details on Symmetric Sweeping:
Thank you to great notes provided by @Hua-Zhou

![](https://cloud.githubusercontent.com/assets/8075494/17649366/f0c9e7da-6201-11e6-8646-27607933d531.png)

![](https://cloud.githubusercontent.com/assets/8075494/17649375/2afe0a1c-6202-11e6-8f99-ed34c580d804.png)
