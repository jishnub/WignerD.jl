using WignerD,PointsOnASphere,TwoPointFunctions,LegendrePolynomials,
OffsetArrays,SphericalHarmonics,SphericalHarmonicArrays,Test

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
	@testset "wignersymbols Wigner3j" begin
		for j1=0:5,j2=0:5
			smin,smax = abs(j1-j2),j1+j2
			for m=-j1:j1,t=max(-smax,-j2+m):min(smax,j2+m)
				W3 = WignerD.Wigner3j(j1,j2,m,t-m)[1:smax-max(abs(t),smin)+1]
				W32 = [WignerD.wigner3j(j1,j2,s,m,t-m) for s=max(abs(t),smin):smax]
				@test W3 ≈ W32
			end
		end
	end
	@testset "wignersymbols CG" begin
		for j1=0:5,j2=0:5
			smin,smax = abs(j1-j2),j1+j2
			for m=-j1:j1,t=max(-smax,-j2+m):min(smax,j2+m)
				CG = WignerD.CG_ℓ₁mℓ₂nst(j1,m,j2,t)[max(abs(t),smin):smax]
				CGW = [WignerD.clebschgordan(j1,m,j2,t-m,s,t) for s=max(abs(t),smin):smax]
				@test CG ≈ CGW
			end
		end
	end
end

@testset "Ylm0" begin
	n = Point2D(π*rand(),2π*rand())
	@test Ylmatrix(OSH(),1,n) ≈ OffsetArray([√(3/8π)*sin(n.θ)cis(-n.ϕ),
										√(3/4π)cos(n.θ),
										-√(3/8π)*sin(n.θ)cis(n.ϕ)],-1:1)
end

@testset "Ylmatrix OSH and GSH" begin
	ℓ = rand(1:10)
	n = Point2D(π/2,0)
	Y1 = Ylmatrix(GSH(),ℓ,n)
	Y2 = Ylmatrix(OSH(),ℓ,n)
	@test Y1[:,0] ≈ Y2
end

@testset "Y1100 explicit" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	@test BiPoSH(OSH(),n1,n2,0,0,1,1) ≈ -√3/4π * cosχ(n1,n2)
end
	
@testset "Yℓℓ_00" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	ℓmax = 10
	Yℓℓ_00 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	P = Pl(cosχ(n1,n2),lmax=ℓmax)
	
	for ℓ in axes(Yℓℓ_00,1)
		Yℓℓ_00[ℓ] = P[ℓ]*(-1)^ℓ * √(2ℓ+1)/4π
	end
	
	YB_00 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	for ℓ in axes(YB_00,1)
		YB_00[ℓ] = BiPoSH(OSH(),n1,n2,0,0,ℓ,ℓ)
	end
	
	@test Yℓℓ_00 ≈ YB_00
end

@testset "Yℓℓ_10" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	ℓmax = 10
	Yℓℓ_10 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	dP = dPl(cosχ(n1,n2),lmax=ℓmax)
	
	for ℓ in axes(Yℓℓ_10,1)
		Yℓℓ_10[ℓ] = dP[ℓ]*im*(-1)^ℓ * √(3*(2ℓ+1)/(ℓ*(ℓ+1)))/4π * ∂ϕ₂cosχ(n1,n2)
	end
	
	YB_10_n1n2 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	YB_10_n2n1 = OffsetArray{ComplexF64}(undef,1:ℓmax)
	for ℓ in 1:ℓmax
		YB_10_n1n2[ℓ] = BiPoSH(OSH(),n1,n2,1,0,ℓ,ℓ)
		YB_10_n2n1[ℓ] = BiPoSH(OSH(),n2,n1,1,0,ℓ,ℓ)
	end
	
	@test Yℓℓ_10 ≈ YB_10_n1n2
	@test YB_10_n1n2 ≈ -YB_10_n2n1
end

@testset "BiPoSH_OSH_10" begin
	n1 = Point2D(π*rand(),2π*rand());
	n2 = Point2D(π*rand(),2π*rand());
	lmax = 10;
	Yℓℓ_10 = zeros(ComplexF64,1:lmax);
	dP = dPl(cosχ(n1,n2),lmax=lmax);
	
	for ℓ in axes(Yℓℓ_10,1)
		Yℓℓ_10[ℓ] = dP[ℓ]*im*(-1)^ℓ * √(3*(2ℓ+1)/(ℓ*(ℓ+1)))/4π * ∂ϕ₂cosχ(n1,n2)
	end
	
	YB_10_n1n2 = zeros(ComplexF64,axes(Yℓℓ_10,1));
	YB_10_n2n1 = zeros(ComplexF64,axes(Yℓℓ_10,1));

	YSH1,YSH2,P,coeff = WignerD.allocate_Y₁Y₂(OSH(),lmax);
	WignerD.compute_YP!(lmax,(n1.θ,n1.ϕ),YSH1,P,coeff);
	WignerD.compute_YP!(lmax,(n2.θ,n2.ϕ),YSH2,P,coeff);

	B = SHVector(LM(1:1,0:0));

	for ℓ in axes(Yℓℓ_10,1)
		
		BiPoSH!(OSH(),n1,n2,B,ℓ,ℓ,YSH1,YSH2,P,coeff,
			compute_Y₁=false,compute_Y₂=false);
		
		YB_10_n1n2[ℓ] = B[(1,0)]

		BiPoSH!(OSH(),n2,n1,B,ℓ,ℓ,YSH2,YSH1,P,coeff,
			compute_Y₁=false,compute_Y₂=false);

		YB_10_n2n1[ℓ] = B[(1,0)]
	end
	
	@test Yℓℓ_10 ≈ YB_10_n1n2
	@test YB_10_n1n2 ≈ -YB_10_n2n1
end

@testset "BiPoSH OSH and GSH" begin
	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)
	SHModes = LM(0:1,0:0)
    B_GSH=BiPoSH(GSH(),n1,n2,SHModes,2,2)
    B_OSH=BiPoSH(OSH(),n1,n2,SHModes,2,2)
    @test B_GSH[:,0,0] ≈ B_OSH
end

@testset "BiPoSH ℓrange" begin
	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)
	SHModes = LM(0,1,0,0)
	ℓ_range = 1:10
	ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes)
    B_all = BiPoSH(OSH(),n1,n2,SHModes,ℓ′ℓ)

    for (ℓ′,ℓ) in ℓ′ℓ
    	B = BiPoSH(OSH(),n1,n2,SHModes,ℓ′,ℓ)
    	@test B_all[(ℓ′,ℓ)] ≈ B
    end
end

@testset "BiPoSH all (l,m) one (l₁,l₂)" begin
	n1 = Point2D(π/2,0);
	n2 = Point2D(π/2,π/3);
    SHModes = LM(0:2)
    ℓ′,ℓ = 1,2
    b = BiPoSH(OSH(),n1,n2,SHModes,ℓ′,ℓ)
    for (s,t) in shmodes(b)
    	@test b[(s,t)] ≈ BiPoSH(OSH(),n1,n2,s,t,ℓ′,ℓ)
    end
end

@testset "BiPoSH ℓrange 2pt" begin
	n1 = Point2D(π/2,0);
	n2 = Point2D(π/2,π/3);
	SHModes = LM(0,2);
	ℓ_range = 0:2;
	ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes);
	function test_and_print_fail(arr,(ℓ′ℓind,ℓ′,ℓ),match_arr)
		@test begin 
    		res = arr[ℓ′ℓind] ≈ match_arr
    		if !res
    			println(ℓ′ℓind," ",ℓ′," ",ℓ)
    			println()
    			display(arr[ℓ′ℓind])
    			println()
    			display(match_arr)
    		end
    		res
    	end
	end
	@testset "all ℓ′" begin
	    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(OSH(),n1,n2,SHModes,ℓ′ℓ)
	    Yℓ′n₁ℓn₂_all_2,Yℓ′n₂ℓn₁_all_2 = BiPoSH_n1n2_n2n1(OSH(),n1,n2,SHModes,ℓ_range)
	    @test shmodes(Yℓ′n₁ℓn₂_all) == ℓ′ℓ
		@test shmodes(Yℓ′n₁ℓn₂_all) == ℓ′ℓ
		@test shmodes(Yℓ′n₂ℓn₁_all_2) == ℓ′ℓ
		@test shmodes(Yℓ′n₂ℓn₁_all_2) == ℓ′ℓ
	    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
	    	Yℓ′n₁ℓn₂ = BiPoSH(OSH(),n1,n2,SHModes,ℓ′,ℓ)
	    	Yℓ′n₂ℓn₁ = BiPoSH(OSH(),n2,n1,SHModes,ℓ′,ℓ)
	    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
	    	test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
	    	test_and_print_fail(Yℓ′n₁ℓn₂_all_2,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
	    	test_and_print_fail(Yℓ′n₂ℓn₁_all_2,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
	    end	    
	end

    @testset "some ℓ′" begin
	    ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes);
	    ℓ′range = l₂_range(ℓ′ℓ)
	    if length(ℓ′range) > 1
		    ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes,ℓ′range[2:end]);
		    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(OSH(),n1,n2,SHModes,ℓ′ℓ)
		    @test shmodes(Yℓ′n₁ℓn₂_all) == ℓ′ℓ
		    @test shmodes(Yℓ′n₂ℓn₁_all) == ℓ′ℓ

		    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
		    	Yℓ′n₁ℓn₂ = BiPoSH(OSH(),n1,n2,SHModes,ℓ′,ℓ)
		    	Yℓ′n₂ℓn₁ = BiPoSH(OSH(),n2,n1,SHModes,ℓ′,ℓ)
		    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
	    		test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
		    end
		end
    end
end