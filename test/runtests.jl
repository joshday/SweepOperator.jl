using SweepOperator
using Base.Test

x = randn(100, 10)
xtx = x'x

A = deepcopy(xtx)
B = deepcopy(xtx)

for j in 1:4
    sweep!(A, j)
    sweep!(A, j, true)
end

@testset "Tests" begin
    @test A ≈ B
end
