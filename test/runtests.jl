using WignerD,PointsOnASphere,TwoPointFunctions,LegendrePolynomials
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
	@testset "allocating" begin
		CG = WignerD.CG_ℓ₁mℓ₂nst(1,1,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6

		CG = WignerD.CG_ℓ₁mℓ₂nst(1,-1,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,-1,1,1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,-1,1,1,1,0) ≈ -1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,-1,1,1,2,0) ≈ 1/√6

		CG = WignerD.CG_ℓ₁mℓ₂nst(1,0,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,0,1,0,0,0) ≈ -1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,0,1,0,1,0) ≈ 0
		@test CG[2] ≈ WignerD.clebschgordan(1,0,1,0,2,0) ≈ √(2/3)
	end
	@testset "non-allocating" begin
		CG = zeros(0:2)
		w3j = zeros(3)
		WignerD.CG_ℓ₁mℓ₂nst!(1,1,1,0,CG,w3j)

		@test CG[0] ≈ WignerD.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6

		WignerD.CG_ℓ₁mℓ₂nst!(1,1,1,0,CG)

		@test CG[0] ≈ WignerD.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6		
	end
end

@testset "Ylm0" begin
	n = Point2D(π*rand(),2π*rand())
	@test Ylmatrix(1,n,n_range=0:0) ≈ OffsetArray(reshape([√(3/8π)*sin(n.θ)cis(-n.ϕ),
										√(3/4π)cos(n.θ),
										-√(3/8π)*sin(n.θ)cis(n.ϕ)],3,1),-1:1,0:0)
end

@testset "Ylmatrix OSH and GSH" begin
	ℓ = rand(1:10)
	n = Point2D(π/2,0)
	Y1 = Ylmatrix(GSH(),ℓ,n)
	Y2 = Ylmatrix(OSH(),ℓ,n)
	@test Y1[:,0] ≈ Y2[:,0]
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
	P = Pl(cosχ(n1,n2),lmax=ℓmax)
	
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
	dP = dPl(cosχ(n1,n2),lmax=ℓmax)
	
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
	b_st = BiPoSH(ℓ,ℓ,0:2ℓ,n1,n2,β=0,γ=0,t=0)
	b_s0 = BiPoSH_s0(ℓ,ℓ,0:2ℓ,0,0,n1,n2)
	@test parent(parent(b_st)) ≈ parent(b_s0)
end

@testset "BiPoSH OSH and GSH" begin
	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)
	SHModes = st(0,1,0,0)
    B_GSH=BiPoSH(GSH(),2,2,SHModes,n1,n2)
    B_OSH=BiPoSH(OSH(),2,2,SHModes,n1,n2)
    @test B_GSH[:,0,0] ≈ B_OSH[:,0,0]
end

@testset "BiPoSH ℓrange" begin
	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)
	SHModes = st(0,1,0,0)
	ℓ_range = 1:10
	ℓ′ℓ = s′s(ℓ_range,SHModes)
    B_all = BiPoSH(OSH(),ℓ_range,SHModes,n1,n2)

    for (ℓ′,ℓ) in ℓ′ℓ
    	B = BiPoSH(OSH(),ℓ′,ℓ,SHModes,n1,n2)
    	@test B_all[:,modeindex(ℓ′ℓ,(ℓ′,ℓ))] ≈ B[:,0,0]
    end
end

@testset "BiPoSH ℓrange 2pt" begin
	n1 = Point2D(π/2,0);
	n2 = Point2D(π/2,π/3);
	SHModes = st(0,10);
	ℓ_range = 1:10;
	ℓ′ℓ = s′s(ℓ_range,SHModes);
    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(OSH(),ℓ_range,SHModes,n1,n2)

    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
    	Yℓ′n₁ℓn₂ = BiPoSH(OSH(),ℓ′,ℓ,SHModes,n1,n2)
    	Yℓ′n₂ℓn₁ = BiPoSH(OSH(),ℓ′,ℓ,SHModes,n2,n1)
    	@test Yℓ′n₁ℓn₂_all[:,ℓ′ℓind] ≈ Yℓ′n₁ℓn₂[:,0,0]
    	@test Yℓ′n₂ℓn₁_all[:,ℓ′ℓind] ≈ Yℓ′n₂ℓn₁[:,0,0]
    end
end