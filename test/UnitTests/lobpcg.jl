using COSMO
using Random, Test, LinearAlgebra

verbosity = 0 # Change to 2 to see LOBPCG's output
rng = MersenneTwister(1)
λ = [1:6; -50.00:-0.00]
n = length(λ)
Q = Matrix(qr(randn(rng, n, n)).Q)
X = Q*Diagonal(λ)*Q'

n_ = Int(n*(n + 1)/2)

function exact_projection(A::Matrix)
    l, V = eigen(Symmetric(A))
    return Symmetric(V*Diagonal(max.(l, 0))*V')
end

function lobpcg_unit_test!(cone, X::Matrix)
    x = Main.COSMO.extract_upper_triangle(X)
    X_proj_exact = exact_projection(X)
    COSMO.project!(x, cone)
    X_proj = Main.COSMO.populate_upper_triangle(x)

    @test norm(X_proj - X_proj_exact) < 1e-5 # This is tolerance is hand-wavy
end

function lobpcg_unit_tests(X::Matrix)
    cone = Main.COSMO.PsdConeTriangleLOBPCG(n_, verbosity=verbosity, max_iter=200)
    @testset "Initial call (exact eigendecomposition)" begin
        lobpcg_unit_test!(cone, X)
        @test cone.exact_projections == 1
        @test cone.lobpcg_iterations == 0
    end

    @testset "Warm starting - already solved to tolerance" begin
        cone.iteration = 10^3
        lobpcg_unit_test!(cone, X + 1e-6*Symmetric(randn(rng, n, n)))
        @test cone.exact_projections == 1
        @test cone.lobpcg_iterations == 1
    end

    @testset "Warm starting - a few iterations required" begin
        cone.iteration = 10^8
        lobpcg_unit_test!(cone, X + 1e-6*Symmetric(randn(rng, n, n)))
        @test cone.exact_projections == 1
        @test cone.lobpcg_iterations >= 3
        @test cone.lobpcg_iterations < 10
    end

    @testset "Warm starting - expand subspace" begin
        cone.iteration = 10^8
        if cone.is_subspace_positive
            lobpcg_unit_test!(cone, X + 3.1*I)
        else
            lobpcg_unit_test!(cone, X - 3.1*I)
        end
        @test size(cone.U, 2) > 6
        @test cone.exact_projections == 1
        @test cone.lobpcg_iterations >= 3
        @test cone.lobpcg_iterations < 50
    end

    @testset "Warm starting - contract subspace" begin
        cone.iteration = 10^8
        if cone.is_subspace_positive
            lobpcg_unit_test!(cone, X - 1.1*I)
        else
            lobpcg_unit_test!(cone, X + 1.1*I)
        end
        @test size(cone.U, 2) < 6
        @test cone.exact_projections == 1
        @test cone.lobpcg_iterations >= 3
        @test cone.lobpcg_iterations < 60
    end
end

@testset "LOBPCG PSD Projection" begin
    @testset "Track Positive Eigenspace" begin
        lobpcg_unit_tests(X)
    end
    @testset "Track Negative Eigenspace" begin
        lobpcg_unit_tests(-X)
    end
end

nothing
