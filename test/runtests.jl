using WignerD,PointsOnASphere,TwoPointFunctions,LegendrePolynomials,
OffsetArrays,SphericalHarmonics,SphericalHarmonicArrays,Test

@testset "cis" begin
	@testset "Equator" begin
	    for α = -10:10
	    	@test cis(α,Equator()) ≈ cis(α*π/2)
	    	@test cis(float(α),Equator()) ≈ cis(α*π/2)
	    end
	end
	@testset "North Pole" begin
	   for α = -10:10
	    	@test cis(α,NorthPole()) ≈ cis(0)
	    	@test cis(float(α),NorthPole()) ≈ cis(0)
	    end
	end
	@testset "South Pole" begin
	   for α = -10:10
	    	@test cis(α,SouthPole()) ≈ cis(α*π)
	    	@test cis(float(α),SouthPole()) ≈ cis(α*π)
	    end
	end
end

@testset "djmatrix_terms" begin
	j = 5
    λ,v = Jy_eigen(j)

    function testapprox(m,n,dj_m_n,dj_m_n_πmθ,dj_n_m,dj_m_n2,dj_m_n_πmθ2,dj_n_m2)
    	@test begin 
    		res = isapprox(dj_m_n,dj_m_n2,atol=1e-14,rtol=sqrt(eps(Float64)))
    		if !res
    			@show m n dj_m_n dj_m_n2
    		end
    		res
    	end
    	@test begin 
    		res = isapprox(dj_m_n_πmθ,dj_m_n_πmθ2,atol=1e-14,rtol=sqrt(eps(Float64)))
    		if !res
    			@show m n dj_m_n_πmθ dj_m_n_πmθ2
    		end
    		res
    	end	
    	@test begin 
    		res = isapprox(dj_n_m,dj_n_m2,atol=1e-14,rtol=sqrt(eps(Float64)))
    		if !res
    			@show m n dj_n_m dj_n_m2
    		end
    		res
    	end
    end
    
    @testset "Equator" begin
        for m in -j:j, n in -j:j
        	dj_m_n,dj_m_n_πmθ,dj_n_m = WignerD.djmatrix_terms(π/2,λ,v,m,n)
        	dj_m_n2,dj_m_n_πmθ2,dj_n_m2 = WignerD.djmatrix_terms(Equator(),λ,v,m,n)

        	testapprox(m,n,dj_m_n,dj_m_n_πmθ,dj_n_m,dj_m_n2,dj_m_n_πmθ2,dj_n_m2)
        end
    end
    @testset "NorthPole" begin
        for m in -j:j, n in -j:j
        	dj_m_n,dj_m_n_πmθ,dj_n_m = WignerD.djmatrix_terms(0,λ,v,m,n)
        	dj_m_n2,dj_m_n_πmθ2,dj_n_m2 = WignerD.djmatrix_terms(NorthPole(),λ,v,m,n)

        	testapprox(m,n,dj_m_n,dj_m_n_πmθ,dj_n_m,dj_m_n2,dj_m_n_πmθ2,dj_n_m2)
        end
    end
    @testset "SouthPole" begin
        for m in -j:j, n in -j:j
        	dj_m_n,dj_m_n_πmθ,dj_n_m = WignerD.djmatrix_terms(π,λ,v,m,n)
        	dj_m_n2,dj_m_n_πmθ2,dj_n_m2 = WignerD.djmatrix_terms(SouthPole(),λ,v,m,n)

        	testapprox(m,n,dj_m_n,dj_m_n_πmθ,dj_n_m,dj_m_n2,dj_m_n_πmθ2,dj_n_m2)
        end
    end
end

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

@testset "Clebsch-Gordan" begin
	@testset "allocating" begin
		CG = WignerD.CG_l₁m₁_l₂m₂_lm(1,1,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6

		CG = WignerD.CG_l₁m₁_l₂m₂_lm(1,-1,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,-1,1,1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,-1,1,1,1,0) ≈ -1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,-1,1,1,2,0) ≈ 1/√6

		CG = WignerD.CG_l₁m₁_l₂m₂_lm(1,0,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,0,1,0,0,0) ≈ -1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,0,1,0,1,0) ≈ 0
		@test CG[2] ≈ WignerD.clebschgordan(1,0,1,0,2,0) ≈ √(2/3)
	end
	@testset "non-allocating" begin
		CG = zeros(0:2)

		WignerD.CG_l₁m₁_l₂m₂_lm!(1,1,1,0,CG)

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
				CG = WignerD.CG_l₁m₁_l₂m₂_lm(j1,m,j2,t)[max(abs(t),smin):smax]
				CGW = [WignerD.clebschgordan(j1,m,j2,t-m,s,t) for s=max(abs(t),smin):smax]
				@test CG ≈ CGW
			end
		end
	end
end

@testset "Y1m0" begin
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

@testset "Ylmatrix special points" begin
    @testset "OSH" begin
        Y1 = Ylmatrix(OSH(),3,(π/2,π/2))
        Y2 = Ylmatrix(OSH(),3,(Equator(),π/2))
        Y3 = Ylmatrix(OSH(),3,(Equator(),Piby2()))
        @test Y1 ≈ Y2
        @test Y1 ≈ Y3
    end
    @testset "GSH" begin
		Y1 = Ylmatrix(GSH(),3,(π/2,π/2))
        Y2 = Ylmatrix(GSH(),3,(Equator(),π/2))
        Y3 = Ylmatrix(GSH(),3,(Equator(),Piby2()))
        @test Y1 ≈ Y2
        @test Y1 ≈ Y3
    end
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
	n1 = Point2D(π/3,0)
	n2 = Point2D(π/3,π/3)
	s_max = 20
	SHModes = LM(0:s_max);
	for j₁ in 0:20, j₂ in 0:20
		WignerD.δ(j₁,j₂,s_max) || continue
	    B_GSH=BiPoSH(GSH(),n1,n2,SHModes,j₂,j₁)
	    B_OSH=BiPoSH(OSH(),n1,n2,SHModes,j₂,j₁)
    	@test begin
    		res = B_GSH[:,0,0] ≈ B_OSH
    		if !res
    			@show (j₁,j₂,s_max) B_GSH[:,0,0] B_OSH
    		end
    		res
    	end
    end
end

@testset "BiPoSH GSH conjugate" begin
    n1 = Point2D(π/3,0)
	n2 = Point2D(π/3,π/3)
	n1 = Point2D(π/3,0)
	n2 = Point2D(π/3,π/3)

	@testset "m=0" begin
		function testconj(l,j₁,j₂)
		    phase = (-1)^(j₁+j₂+l)
		    b = BiPoSH(GSH(),n1,n2,l,0,j₁,j₂)
		    for α₂ = -1:1, α₁ = -1:1
		    	@test begin
		    		res = b[α₁,α₂] ≈ phase * (-1)^(α₁+α₂) * conj(b[-α₁,-α₂])
		    		if !res
		    			@show (l,j₁,j₂,α₁,α₂) b[α₁,α₂] b[-α₁,-α₂]
		    		end
		    		res
		    	end
		    end
		end
		for j₁ = 1:4, j₂ = 1:4, l = abs(j₁ - j₂):j₁+j₂
			testconj(l,j₁,j₂)
		end
	end
	@testset "m and -m" begin
	    function testconj(l,m,j₁,j₂)
		    
		    phase = (-1)^(j₁+j₂+l+m)
		    
		    b1 = BiPoSH(GSH(),n1,n2,l,m,j₁,j₂)
		    b2 = BiPoSH(GSH(),n1,n2,l,-m,j₁,j₂)

		    for α₂ = -1:1, α₁ = -1:1
		    	@test begin
		    		res = isapprox(b1[α₁,α₂], phase * (-1)^(α₁+α₂) * conj(b2[-α₁,-α₂]), 
		    			atol=1e-15, rtol= sqrt(eps(Float64)))
		    		if !res
		    			@show (l,m,j₁,j₂,α₁,α₂) b1[α₁,α₂] b2[-α₁,-α₂]
		    		end
		    		res
		    	end
		    end
		end
		for j₁ = 1:4, j₂ = 1:4, l = abs(j₁ - j₂):j₁+j₂, m = -l:l
			testconj(l,m,j₁,j₂)
		end
	end
end

@testset "BiPoSH ℓrange" begin
	n1 = Point2D(π/3,0)
	n2 = Point2D(π/3,π/3)
	SHModes = LM(0:5)
	ℓ_range = 1:10
	ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes)

	@testset "OSH" begin
	    B_all = BiPoSH(OSH(),n1,n2,SHModes,ℓ′ℓ)

	    for (ℓ′,ℓ) in ℓ′ℓ
	    	B = BiPoSH(OSH(),n1,n2,SHModes,ℓ′,ℓ)
	    	@test B_all[(ℓ′,ℓ)] ≈ B
	    end
	end
	@testset "GSH" begin
	    B_all = BiPoSH(GSH(),n1,n2,SHModes,ℓ′ℓ)

	    for (ℓ′,ℓ) in ℓ′ℓ
	    	B = BiPoSH(GSH(),n1,n2,SHModes,ℓ′,ℓ)
	    	@test B_all[(ℓ′,ℓ)] ≈ B
	    end
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

	function test_and_print_fail(arr,(ℓ′ℓind,ℓ′,ℓ),match_arr)
		@test begin
			res = false
			try
    			res = arr[ℓ′ℓind] ≈ match_arr
    		catch
    			@show ℓ ℓ′ axes(arr[ℓ′ℓind]) axes(match_arr)
    			rethrow()
    		end

    		if !res
    			@show (ℓ′ℓind,ℓ′,ℓ)
    			println()
    			@show arr[ℓ′ℓind]
    			println()
    			@show match_arr
    		end
    		res
    	end
	end

	@testset "OSH" begin
		SHModes = LM(0,5);
		ℓ_range = 0:5;
		ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes);

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

	@testset "GSH" begin
		SHModes = LM(0,5);
		ℓ_range = 0:5;
		ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes);
		
		function testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			# We test for Yʲ²ʲ¹ₗₘ_α₂α₁_n₂n₁ = (-1)^(j₁+j₂+l) * Yʲ¹ʲ²ₗₘ_α₁α₂_n₁n₂
	    	for (ℓ′,ℓ) in Iterators.product(ℓ_common,ℓ_common)
	    		Yℓ′n₁ℓn₂ = Yℓ′n₁ℓn₂_all[(ℓ′,ℓ)]
	    		Yℓn₂ℓ′n₁ = Yℓ′n₂ℓn₁_all[(ℓ,ℓ′)]
	    		for (s,t) in shmodes(Yℓ′n₁ℓn₂)
	    			phase = (-1)^(ℓ′ + ℓ + s)
	    			Yℓ′n₁ℓn₂_st = Yℓ′n₁ℓn₂[(s,t),:,:]
	    			Yℓn₂ℓ′n₁_st = Yℓn₂ℓ′n₁[(s,t),:,:]
    				@test begin
    					res = Yℓ′n₁ℓn₂_st ≈ phase .* permutedims(Yℓn₂ℓ′n₁_st)
    					if !res
    						@show (s,t) Yℓ′n₁ℓn₂_st Yℓn₂ℓ′n₁_st
    					end
    					res
    				end
	    		end
	    	end
	    end

	    function testOSH(Yℓ′n₁ℓn₂_all,Yℓ′n₁ℓn₂_all_OSH)
	    	@test shmodes(Yℓ′n₁ℓn₂_all) == shmodes(Yℓ′n₁ℓn₂_all_OSH)
	    	for j₂j₁ind in axes(Yℓ′n₁ℓn₂_all,1)
	    		YGSH = Yℓ′n₁ℓn₂_all[j₂j₁ind]
	    		YOSH = Yℓ′n₁ℓn₂_all_OSH[j₂j₁ind]
	    		@test shmodes(YGSH) == shmodes(YOSH)
	    		@test YGSH[:,0,0] ≈ YOSH
	    	end
	    end

		@testset "all ℓ′" begin
		    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(GSH(),n1,n2,SHModes,ℓ′ℓ)
		    Yℓ′n₁ℓn₂_all_2,Yℓ′n₂ℓn₁_all_2 = BiPoSH_n1n2_n2n1(GSH(),n1,n2,SHModes,ℓ_range)
		    
		    @test Yℓ′n₁ℓn₂_all == Yℓ′n₁ℓn₂_all_2
		    @test Yℓ′n₂ℓn₁_all == Yℓ′n₂ℓn₁_all_2

		    @test shmodes(Yℓ′n₁ℓn₂_all) == ℓ′ℓ
			@test shmodes(Yℓ′n₁ℓn₂_all) == ℓ′ℓ
			@test shmodes(Yℓ′n₂ℓn₁_all_2) == ℓ′ℓ
			@test shmodes(Yℓ′n₂ℓn₁_all_2) == ℓ′ℓ
			@testset "match with B(ℓ′,ℓ)" begin
			    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
			    	Yℓ′n₁ℓn₂ = BiPoSH(GSH(),n1,n2,SHModes,ℓ′,ℓ)
			    	Yℓ′n₂ℓn₁ = BiPoSH(GSH(),n2,n1,SHModes,ℓ′,ℓ)
			    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
			    	test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
			    end
			end
			@testset "match flip" begin
				ℓ′_range = l₂_range(ℓ′ℓ)
				ℓ_common = intersect(ℓ′_range,ℓ_range)
			    testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			end
			@testset "compare with OSH" begin
			    Yℓ′n₁ℓn₂_all_OSH,Yℓ′n₂ℓn₁_all_OSH = BiPoSH_n1n2_n2n1(OSH(),n1,n2,SHModes,ℓ′ℓ)
			    testOSH(Yℓ′n₁ℓn₂_all,Yℓ′n₁ℓn₂_all_OSH)
			    testOSH(Yℓ′n₂ℓn₁_all,Yℓ′n₂ℓn₁_all_OSH)
			end
		end

	    @testset "some ℓ′" begin
		    ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes);
		    ℓ′range = l₂_range(ℓ′ℓ)
		    if length(ℓ′range) > 1
			    ℓ′ℓ = L₂L₁Δ(ℓ_range,SHModes,ℓ′range[2:end]);
			    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(GSH(),n1,n2,SHModes,ℓ′ℓ)
			    @test shmodes(Yℓ′n₁ℓn₂_all) == ℓ′ℓ
			    @test shmodes(Yℓ′n₂ℓn₁_all) == ℓ′ℓ

			    @testset "match with B(ℓ′,ℓ)" begin
				    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
				    	Yℓ′n₁ℓn₂ = BiPoSH(GSH(),n1,n2,SHModes,ℓ′,ℓ)
				    	Yℓ′n₂ℓn₁ = BiPoSH(GSH(),n2,n1,SHModes,ℓ′,ℓ)
				    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
			    		test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
				    end
				end
			end
			@testset "match flip" begin
				ℓ′_range = l₂_range(ℓ′ℓ)
				ℓ_common = intersect(ℓ′_range,ℓ_range)
			    testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			end
			@testset "compare with OSH" begin
			    Yℓ′n₁ℓn₂_all_OSH,Yℓ′n₂ℓn₁_all_OSH = BiPoSH_n1n2_n2n1(OSH(),n1,n2,SHModes,ℓ′ℓ)
			    testOSH(Yℓ′n₁ℓn₂_all,Yℓ′n₁ℓn₂_all_OSH)
			    testOSH(Yℓ′n₂ℓn₁_all,Yℓ′n₂ℓn₁_all_OSH)
			end
	    end
	end
end