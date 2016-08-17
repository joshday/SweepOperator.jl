using SweepOperator

# necessary to support 0.4 and 0.5
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end


# setup
n, p = 1000, 10
x = randn(n, p)
xtx = x'x



@testset "Sweep One By One" begin
    A = deepcopy(xtx)
    B = deepcopy(xtx)
    for j in 1:p
        sweep!(A, j)
        sweep!(A, j, true)
    end
    @test A ≈ B

    A = deepcopy(xtx)
    B = deepcopy(xtx)
    for j in 1:p
        sweep!(A, j, true)
    end
    for j in 1:p
        sweep!(A, j, true)
    end
    @test A ≈ B
end

@testset "Sweep All" begin
    A = deepcopy(xtx)
    B = deepcopy(xtx)
    sweep!(A, 1:p)
    sweep!(A, 1:p, true)
    @test A ≈ B
end

@testset "Linear Regression" begin
    x = randn(n, p)
    y = x * collect(1.:p) + randn(n)
    xy = [x y]
    xytxy = xy'xy
    sweep!(xytxy, 1:p)
    @test xytxy[1:p, end] ≈ x\y
end
