using WignerD, Test
using WignerD: NorthPole, SouthPole, Equator
using Aqua
using HalfIntegers
using OffsetArrays
using LinearAlgebra
using Rotations
using StaticArrays

@testset "project quality" begin
    Aqua.test_all(WignerD, ambiguities = false)
end

@testset "trigonometric functions for special points" begin
	@testset "Equator" begin
        @test promote_rule(Equator, Float64) == Float64
	    for α = -10:10
	    	@test (@inferred WignerD._sincos(α, Equator()))[1] ≈ sincos(α*π/2)[1] atol=1e-14 rtol=1e-8
	    	@test (@inferred WignerD._sincos(α, Equator()))[2] ≈ sincos(α*π/2)[2] atol=1e-14 rtol=1e-8
            @test (@inferred WignerD._sincos(half(α), Equator()))[1] ≈ sincos(half(α)*π/2)[1] atol=1e-14 rtol=1e-8
            @test (@inferred WignerD._sincos(half(α), Equator()))[2] ≈ sincos(half(α)*π/2)[2] atol=1e-14 rtol=1e-8
            @test sincos(α*Equator()) == sincos(α*π/2)
	    end
    	@test @inferred cos(Equator()) == 0
    	@test @inferred sin(Equator()) == 1
    	@test @inferred float(Equator()) == π/2
    	@test @inferred AbstractFloat(Equator()) == π/2
    	@test @inferred Float64(Equator()) == π/2
	end
    @testset "NorthPole" begin
        @test @inferred cos(NorthPole()) == 1
        @test @inferred sin(NorthPole()) == 0
        @test @inferred float(NorthPole()) == 0
        @test @inferred AbstractFloat(NorthPole()) == 0
        @test @inferred Float64(NorthPole()) == 0
    end
    @testset "SouthPole" begin
        @test @inferred cos(SouthPole()) == -1
        @test @inferred sin(SouthPole()) == 0
        @test @inferred float(SouthPole()) == float(pi)
        @test @inferred AbstractFloat(SouthPole()) == float(pi)
        @test @inferred Float64(SouthPole()) == float(pi)
    end
end

@testset "wignerdjmn" begin

    function testapprox(m, n, dj_m_n, dj_m_n2)
    	@test begin
    		res = isapprox(dj_m_n, dj_m_n2, atol=1e-14, rtol=sqrt(eps(Float64)))
    		if !res
    			@show m n dj_m_n dj_m_n2
    		end
    		res
    	end
    end

	function test_special(j, Jy...)
        @testset "j = $j" begin
            @testset "Equator" begin
                for m in -j:j, n in -j:j
                    dj_m_n = @inferred WignerD.wignerdjmn(j, m, n, π/2, Jy...)
                    dj_m_n2 = @inferred WignerD.wignerdjmn(j, m, n, Equator(), Jy...)
                    testapprox(m, n, dj_m_n, dj_m_n2)
                end
            end
            @testset "NorthPole" begin
                for m in -j:j, n in -j:j
                    dj_m_n = @inferred WignerD.wignerdjmn(j, m, n, 0, Jy...)
                    dj_m_n2 = @inferred WignerD.wignerdjmn(j, m, n, NorthPole(), Jy...)
                    testapprox(m, n, dj_m_n, dj_m_n2)
                end
            end
            @testset "SouthPole" begin
                for m in -j:j, n in -j:j
                    dj_m_n = @inferred WignerD.wignerdjmn(j, m, n, π, Jy...)
                    dj_m_n2 = @inferred WignerD.wignerdjmn(j, m, n, SouthPole(), Jy...)
                    testapprox(m, n, dj_m_n, dj_m_n2)
                end
            end
        end
    end

    for j in 0:3
        test_special(j)
        Jy = zeros(ComplexF64, 2j+1, 2j+1)
        test_special(j, Jy)
    end
    for j in half(1):1:half(7)
        test_special(j)
        Jy = zeros(ComplexF64, Int(2j+1), Int(2j+1))
        test_special(j, Jy)
    end

    @testset "wignerDjmn" begin
        Threads.@threads for j in 0:3
            Jy = zeros(ComplexF64, 2j+1, 2j+1)
            for β in LinRange(0, pi, 10), m in -j:j, n in -j:j
                Djmn = WignerD.wignerDjmn(j, m, n, 0, β, 0)
                Djmn2 = WignerD.wignerDjmn(j, m, n, 0, β, 0, Jy)
                @test Djmn == Djmn2
                djmn = WignerD.wignerdjmn(j, m, n, β)
                @test isapprox(Djmn, djmn, atol = 1e-14, rtol = sqrt(eps(Float64)))
            end
        end
        Threads.@threads for j in half(1):1:half(7)
            Jy = zeros(ComplexF64, Int(2j+1), Int(2j+1))
            for β in LinRange(0, pi, 10), m in -j:j, n in -j:j
                Djmn = WignerD.wignerDjmn(j, m, n, 0, β, 0)
                Djmn2 = WignerD.wignerDjmn(j, m, n, 0, β, 0, Jy)
                @test Djmn == Djmn2
                djmn = WignerD.wignerdjmn(j, m, n, β)
                @test isapprox(Djmn, djmn, atol = 1e-14, rtol = sqrt(eps(Float64)))
            end
        end
    end
end

function testwignerd(testelements, j, θ)
    d = wignerd(j, θ)
    testelements(WignerD._offsetmatrix(j, d), θ)
    @test wignerd(HalfInt(j), θ) ≈ d atol=1e-14 rtol=1e-8
    @test isapprox(tr(d), sum(cos(m*θ) for m in -j:j), atol = 1e-14, rtol = sqrt(eps(Float64)))
    @test det(d) ≈ 1
end

@testset "d0_mn(θ)" begin
    function test(d, θ)
        @test isapprox(d[0, 0], 1, atol=1e-14, rtol=1e-8)
    end

    n = 100
    j = 0

    Threads.@threads for θ in LinRange(0, π, n)
        testwignerd(test, j, θ)
    end

    @testset "Equator" begin
        testwignerd(test, j, Equator())
    end

    @testset "NorthPole" begin
        testwignerd(test, j, NorthPole())
    end

    @testset "SouthPole" begin
        testwignerd(test, j, SouthPole())
    end
end

@testset "d1/2_mn(θ)" begin
    function test(d, θ)
        @test isapprox(d[-1/2, -1/2], cos(θ/2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[1/2, -1/2], -sin(θ/2), atol=1e-14, rtol=1e-8)

        @test isapprox(d[-1/2, 1/2], sin(θ/2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[1/2, 1/2], cos(θ/2), atol=1e-14, rtol=1e-8)
    end

    n = 100
    j = half(1)

    Threads.@threads for θ in LinRange(0, π, n)
        testwignerd(test, j, θ)
    end

    @testset "Equator" begin
        testwignerd(test, j, Equator())
    end

    @testset "NorthPole" begin
        testwignerd(test, j, NorthPole())
    end

    @testset "SouthPole" begin
        testwignerd(test, j, SouthPole())
    end
end

@testset "d1_mn(θ)" begin
	function test(d, θ)
		@test isapprox(d[-1, -1], (1+cos(θ))/2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 0, -1], -sin(θ)/√2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 1, -1], (1-cos(θ))/2, atol=1e-14, rtol=1e-8)

		@test isapprox(d[-1, 0],  sin(θ)/√2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 0, 0],  cos(θ), atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 1, 0], -sin(θ)/√2, atol=1e-14, rtol=1e-8)

		@test isapprox(d[-1, 1], (1-cos(θ))/2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 0, 1],  sin(θ)/√2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1, 1], (1+cos(θ))/2, atol=1e-14, rtol=1e-8)
	end

	n = 100
    j = 1

	Threads.@threads for θ in LinRange(0, π, n)
		testwignerd(test, j, θ)
	end

	@testset "Equator" begin
        testwignerd(test, j, Equator())
    end

    @testset "NorthPole" begin
        testwignerd(test, j, NorthPole())
    end

    @testset "SouthPole" begin
        testwignerd(test, j, SouthPole())
    end
end

@testset "d3/2_mn(θ)" begin
    function test(d, θ)
        @test isapprox(d[-3/2, -3/2],  cos(θ/2)^3, atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1/2, -3/2], -√3*sin(θ/2)cos(θ/2)^2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1/2, -3/2],  √3*sin(θ/2)^2*cos(θ/2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 3/2, -3/2],  -sin(θ/2)^3, atol=1e-14, rtol=1e-8)

        @test isapprox(d[-3/2, -1/2],  √3*sin(θ/2)cos(θ/2)^2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1/2, -1/2],  cos(θ/2)*(3cos(θ/2)^2 - 2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1/2, -1/2],  sin(θ/2)*(3sin(θ/2)^2 - 2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 3/2, -1/2],  √3*sin(θ/2)^2*cos(θ/2), atol=1e-14, rtol=1e-8)

        @test isapprox(d[-3/2,  1/2],  √3*sin(θ/2)^2*cos(θ/2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1/2,  1/2],  -sin(θ/2)*(3sin(θ/2)^2 - 2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1/2,  1/2],  cos(θ/2)*(3cos(θ/2)^2 - 2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 3/2,  1/2], -√3*sin(θ/2)cos(θ/2)^2, atol=1e-14, rtol=1e-8)

        @test isapprox(d[-3/2,  3/2],  sin(θ/2)^3, atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1/2,  3/2],  √3*sin(θ/2)^2*cos(θ/2), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1/2,  3/2],  √3*sin(θ/2)cos(θ/2)^2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 3/2,  3/2],  cos(θ/2)^3, atol=1e-14, rtol=1e-8)
    end

    n = 100
    j = 3/2

    Threads.@threads for θ in LinRange(0, π, n)
        testwignerd(test, j, θ)
    end

    @testset "Equator" begin
        testwignerd(test, j, Equator())
    end

    @testset "NorthPole" begin
        testwignerd(test, j, NorthPole())
    end

    @testset "SouthPole" begin
        testwignerd(test, j, SouthPole())
    end
end

@testset "d2_mn(θ)" begin
	function test(d, θ)
		@test isapprox(d[-2, -2],  (1+cos(θ))^2/4, atol=1e-14, rtol=1e-8)
		@test isapprox(d[-1, -2], -sin(θ)*(1+cos(θ))/2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 0, -2],  1/2*√(3/2)*sin(θ)^2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 1, -2], -sin(θ)*(1-cos(θ))/2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 2, -2],  (1-cos(θ))^2/4, atol=1e-14, rtol=1e-8)

        @test isapprox(d[-2, -1],  sin(θ)*(1+cos(θ))/2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1, -1],  (2cos(θ)^2+cos(θ)-1)/2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 0, -1], -√(3/2)*sin(θ)*cos(θ), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1, -1], -(2cos(θ)^2-cos(θ)-1)/2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 2, -1], -sin(θ)*(1-cos(θ))/2, atol=1e-14, rtol=1e-8)

        @test isapprox(d[-2, 0],  1/2*√(3/2)*sin(θ)^2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1, 0],  √(3/2)*sin(θ)*cos(θ), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 0, 0],  1/2*(3cos(θ)^2-1), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1, 0], -√(3/2)*sin(θ)*cos(θ), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 2, 0],  1/2*√(3/2)*sin(θ)^2, atol=1e-14, rtol=1e-8)

        @test isapprox(d[-2, 1],  sin(θ)*(1-cos(θ))/2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[-1, 1], -(2cos(θ)^2-cos(θ)-1)/2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 0, 1],  √(3/2)*sin(θ)*cos(θ), atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 1, 1],  (2cos(θ)^2+cos(θ)-1)/2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 2, 1], -sin(θ)*(1+cos(θ))/2, atol=1e-14, rtol=1e-8)

		@test isapprox(d[-2, 2],  (1-cos(θ))^2/4, atol=1e-14, rtol=1e-8)
		@test isapprox(d[-1, 2],  sin(θ)*(1-cos(θ))/2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 0, 2],  1/2*√(3/2)*sin(θ)^2, atol=1e-14, rtol=1e-8)
		@test isapprox(d[ 1, 2],  sin(θ)*(1+cos(θ))/2, atol=1e-14, rtol=1e-8)
        @test isapprox(d[ 2, 2],  (1+cos(θ))^2/4, atol=1e-14, rtol=1e-8)
	end

    n = 100
	j = 2

    Threads.@threads for θ in LinRange(0, π, n)
        testwignerd(test, j, θ)
    end

    @testset "Equator" begin
        testwignerd(test, j, Equator())
    end

    @testset "NorthPole" begin
        testwignerd(test, j, NorthPole())
    end

    @testset "SouthPole" begin
        testwignerd(test, j, SouthPole())
    end
end

@testset "symmetry" begin
    for β in LinRange(0, pi, 10), j in 0:half(1):5
        djβ = wignerd(j, β)
        djβ_offset = WignerD._offsetmatrix(j, djβ)
        @test wignerd(j, -β)' ≈ djβ
        for k in 1:4
            @test wignerd(j, β + 2pi * k) ≈ (-1)^(2j*k) * djβ
            @test wignerd(j, β - 2pi * k) ≈ (-1)^(2j*k) * djβ

            dj2np1pi = WignerD._offsetmatrix(j, wignerd(j, β + (2k + 1)*pi))
            for m in -j:j, n in -j:j
                @test isapprox(dj2np1pi[m, n], (-1)^((2k+1)*j - n)*djβ_offset[m, -n], atol =1e-14, rtol = 1e-8)
            end
            dj2np1pi = WignerD._offsetmatrix(j, wignerd(j, β - (2k + 1)*pi))
            for m in -j:j, n in -j:j
                @test isapprox(dj2np1pi[m, n], (-1)^(-(2k+1)*j - n)*djβ_offset[m, -n], atol =1e-14, rtol = 1e-8)
            end
        end
        djpimβ = WignerD._offsetmatrix(j, wignerd(j, π - β))
        for m in -j:j, n in -j:j
            @test isapprox(djpimβ[m, n], (-1)^(j + m) * djβ_offset[m, -n], atol = 1e-14, rtol = 1e-8)
            @test isapprox(djpimβ[m, n], (-1)^(j - n) * djβ_offset[-m, n], atol = 1e-14, rtol = 1e-8)
        end
    end
end

@testset "WignerD" begin
    Threads.@threads for β in LinRange(0, pi, 10)
        for j in 0:half(1):3
            d = wignerd(j, β)
            D = wignerD(j, 0, β, 0)
            @test d == D
            for α in LinRange(0, 2pi, 10), γ in LinRange(0, 2pi, 10)
                D = wignerD(j, α, β, γ)
                D_w = WignerD._offsetmatrix(j, D)
                for n in -j:j, m in -j:j
                    @test isapprox(D_w[m, n], WignerD.wignerDjmn(j, m, n, α, β, γ), atol=1e-14, rtol = sqrt(eps(Float64)))
                end
                @test det(D) ≈ 1
                @test D' * D ≈ Diagonal(ones(Int(2j+1)))

                Dinv = wignerD(j, -γ, -β, -α)
                @test inv(D) ≈ Dinv
                @test D' ≈ Dinv
            end
        end
    end
end

@testset "WignerMatrix" begin
    w = WignerD._offsetmatrix(0.5, wignerd(0.5, 0))
    @test IndexStyle(w) == IndexStyle(parent(w))
    @test size(w) == (2,2)
end

@testset "comparison with Cartesian rotations" begin
    # The spherical covariant basis transforms through D¹*
    # The Cartesian basis transforms through RotZYZ
    # The two may be related through U * RotZYZ(α,β,γ) * U' == D¹*(α,β,γ)
    # where U is the matrix that converts from Cartesian to spherical covariant bases
    # U * [ex, ey, ez] == [χ₋₁, χ₀, χ₊₁]
    U = SMatrix{3,3}(
        [
            1/√2   -im/√2   0
            0       0       1
            -1/√2  -im/√2   0
        ]
    )
    j = 1
    Jy = zeros(ComplexF64, 3, 3)
    D = zeros(ComplexF64, 3, 3)
    for α in LinRange(0, 2pi, 10), β in LinRange(0, 2pi, 10), γ in LinRange(0, 2pi, 10)
        wignerD!(D, j, α, β, γ)
        conj!(D)
        @test D ≈ U * RotZYZ(α, β, γ) * U'
    end
end
