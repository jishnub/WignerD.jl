using WignerD,PointsOnASphere,TwoPointFunctions,Legendre
using OffsetArrays
using Test

@testset "d1_mn(θ)" begin
	θ = π*rand()
	d1 = djmatrix(1,θ)
	
	@test d1[1,1] ≈ (1+cos(θ))/2
	@test d1[1,0] ≈ -sin(θ)/√2
	@test d1[1,-1] ≈ (1-cos(θ))/2

	@test d1[0,1] ≈ sin(θ)/√2
	@test d1[0,0] ≈ cos(θ)
	@test d1[0,-1] ≈ -sin(θ)/√2

	@test d1[-1,1] ≈ (1-cos(θ))/2
	@test d1[-1,0] ≈ sin(θ)/√2
	@test d1[-1,-1] ≈ (1+cos(θ))/2
end

@testset "Wigner3j" begin
	@test WignerD.Wigner3j(1,1,1,-1) ≈ [1/√3,1/√6,1/√30]
	@test WignerD.Wigner3j(2,2,1,-1) ≈ [-1/√5,-1/√30,1/√70,√(2/35),2*√(2/35)/3]
end

@testset "Clebsch-Gordan" begin
	@test WignerD.CG_ℓ₁mℓ₂nst(1,1,1)[0] ≈ WignerD.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
	@test WignerD.CG_ℓ₁mℓ₂nst(1,1,1)[1] ≈ WignerD.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
	@test WignerD.CG_ℓ₁mℓ₂nst(1,1,1)[2] ≈ WignerD.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6

	@test WignerD.CG_ℓ₁mℓ₂nst(1,-1,1)[0] ≈ WignerD.clebschgordan(1,-1,1,1,0,0) ≈ 1/√3
	@test WignerD.CG_ℓ₁mℓ₂nst(1,-1,1)[1] ≈ WignerD.clebschgordan(1,-1,1,1,1,0) ≈ -1/√2
	@test WignerD.CG_ℓ₁mℓ₂nst(1,-1,1)[2] ≈ WignerD.clebschgordan(1,-1,1,1,2,0) ≈ 1/√6

	@test WignerD.CG_ℓ₁mℓ₂nst(1,0,1)[0] ≈ WignerD.clebschgordan(1,0,1,0,0,0) ≈ -1/√3
	@test WignerD.CG_ℓ₁mℓ₂nst(1,0,1)[1] ≈ WignerD.clebschgordan(1,0,1,0,1,0) ≈ 0
	@test WignerD.CG_ℓ₁mℓ₂nst(1,0,1)[2] ≈ WignerD.clebschgordan(1,0,1,0,2,0) ≈ √(2/3)
end

@testset "Ylm0" begin
	n = Point2D(π*rand(),2π*rand())
	@test Ylmatrix(1,n,n_range=0:0) ≈ OffsetArray(reshape([√(3/8π)*sin(n.θ)cis(-n.ϕ),
										√(3/4π)cos(n.θ),
										-√(3/8π)*sin(n.θ)cis(n.ϕ)],3,1),-1:1,0:0)
end

@testset "Y1100 explicit" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	@test BiPoSH_s0(1,1,0,0,0,n1,n2)[0,0,0] ≈ -√3/4π * cosχ(n1,n2)
end
	
@testset "Yℓℓ_00" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	ℓmax = 20
	Yℓℓ_00 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	P = Pl(cosχ(n1,n2),ℓmax=ℓmax)
	
	for ℓ in axes(Yℓℓ_00,1)
		Yℓℓ_00[ℓ] = P[ℓ]*(-1)^ℓ * √(2ℓ+1)/4π
	end
	
	YB_00 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	for ℓ in axes(YB_00,1)
		YB_00[ℓ] = BiPoSH_s0(ℓ,ℓ,0,0,0,n1,n2)[0,0,0]
	end
	
	@test Yℓℓ_00 ≈ YB_00
	
end

@testset "Yℓℓ_10" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	ℓmax = 20
	Yℓℓ_10 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	dP = dPl(cosχ(n1,n2),ℓmax=ℓmax)
	
	for ℓ in axes(Yℓℓ_10,1)
		Yℓℓ_10[ℓ] = dP[ℓ]*im*(-1)^ℓ * √(3*(2ℓ+1)/(ℓ*(ℓ+1)))/4π * ∂ϕ₂cosχ(n1,n2)
	end
	
	YB_10_n1n2 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	YB_10_n2n1 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	for ℓ in 1:ℓmax
		YB_10_n1n2[ℓ] = BiPoSH_s0(ℓ,ℓ,1,0,0,n1,n2)[1]
		YB_10_n2n1[ℓ] = BiPoSH_s0(ℓ,ℓ,1,0,0,n2,n1)[1]
	end
	
	@test Yℓℓ_10 ≈ YB_10_n1n2
	@test YB_10_n1n2 ≈ -YB_10_n2n1
end

@testset "BiPoSH t=0" begin
	ℓ = rand(1:30)
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	b_st = BiPoSH(ℓ,ℓ,0:2ℓ,n1,n2,β=0,γ=0,t=0:0)
	b_s0 = BiPoSH_s0(ℓ,ℓ,0:2ℓ,0,0,n1,n2)
	@test parent(parent(b_st)) ≈ parent(b_s0)
end