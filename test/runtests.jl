using SweepOperator, LinearAlgebra, Test

# setup
n, p = 1000, 10
x = randn(n, p)
xtx = x'x

@testset "Sweep One By One" begin
    A = copy(xtx)
    B = copy(xtx)
    for j in 1:p
        sweep!(A, j)
        sweep!(A, j, true)
    end
    @test UpperTriangular(A) ≈ UpperTriangular(B)

    A = copy(xtx)
    B = copy(xtx)
    for j in 1:p
        sweep!(A, j)
    end
    for j in 1:p
        sweep!(A, j, true)
    end
    @test UpperTriangular(A) ≈ UpperTriangular(B)
end

@testset "Sweep All" begin
    A = copy(xtx)
    B = copy(xtx)
    sweep!(A, 1:p)
    sweep!(A, 1:p, true)
    @test A ≈ B
end

@testset "UpperTriangular" begin
    A = UpperTriangular(copy(xtx))
    B = UpperTriangular(copy(xtx))
    sweep!(A, 1:p)
    sweep!(A, 1:p, true)
    @test A ≈ B
end

@testset "Hermitian/Symmetric" begin
    A = Hermitian(copy(xtx))
    B = Symmetric(copy(xtx))
    sweep!(A, 1:p)
    sweep!(A, 1:p, true)
    sweep!(B, 1:p)
    sweep!(B, 1:p, true)
    @test A ≈ B ≈ xtx
end

@testset "Linear Regression" begin
    y = x * collect(1.:p) + randn(n)
    xy = [x y]
    xytxy = xy'xy
    sweep!(xytxy, 1:p)
    @test xytxy[1:p, end] ≈ x\y
end
