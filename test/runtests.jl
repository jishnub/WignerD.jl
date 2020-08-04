using WignerD,PointsOnASphere,TwoPointFunctions,LegendrePolynomials,
OffsetArrays,SphericalHarmonics,SphericalHarmonicModes,
SphericalHarmonicArrays,Test

import SphericalHarmonicArrays: shmodes

@testset "trigonometric functions for special points" begin
	@testset "Equator" begin
		@test one(Equator()) == 1
	    for α = -10:10
	    	@test cis(α,Equator()) ≈ cis(α*π/2)
	    	@test cis(float(α),Equator()) ≈ cis(α*π/2)
	    	@test cos(Equator()) == 0
	    	@test sin(Equator()) == 1
	    	@test csc(Equator()) == 1
	    	@test float(Equator()) == π/2
	    	@test AbstractFloat(Equator()) == π/2
	    	@test Float64(Equator()) == π/2
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
	A = zeros(ComplexF64,2j+1,2j+1)
    λ,v = WignerD.Jy_eigen!(j,A)

    function testapprox(m,n,dj_m_n,dj_m_n2)
    	@test begin 
    		res = isapprox(dj_m_n,dj_m_n2,atol=1e-14,rtol=sqrt(eps(Float64)))
    		if !res
    			@show m n dj_m_n dj_m_n2
    		end
    		res
    	end
    end
    
    @testset "Equator" begin
        for m in -j:j, n in -j:j
        	dj_m_n = WignerD.djmatrix_terms(π/2,λ,v,m,n)
        	dj_m_n2 = WignerD.djmatrix_terms(Equator(),λ,v,m,n)

        	testapprox(m,n,dj_m_n,dj_m_n2)
        end
    end
    @testset "NorthPole" begin
        for m in -j:j, n in -j:j
        	dj_m_n = WignerD.djmatrix_terms(0,λ,v,m,n)
        	dj_m_n2 = WignerD.djmatrix_terms(NorthPole(),λ,v,m,n)

        	testapprox(m,n,dj_m_n,dj_m_n2)
        end
    end
    @testset "SouthPole" begin
        for m in -j:j, n in -j:j
        	dj_m_n = WignerD.djmatrix_terms(π,λ,v,m,n)
        	dj_m_n2 = WignerD.djmatrix_terms(SouthPole(),λ,v,m,n)

        	testapprox(m,n,dj_m_n,dj_m_n2)
        end
    end
end

@testset "d1_mn(θ)" begin
	function test(d,θ)
		@test isapprox(d[1,1],(1+cos(θ))/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[1,0],-sin(θ)/√2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[1,-1],(1-cos(θ))/2,atol=1e-14,rtol=1e-8)

		@test isapprox(d[0,1],sin(θ)/√2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[0,0],cos(θ),atol=1e-14,rtol=1e-8)
		@test isapprox(d[0,-1],-sin(θ)/√2,atol=1e-14,rtol=1e-8)

		@test isapprox(d[-1,1],(1-cos(θ))/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-1,0],sin(θ)/√2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-1,-1],(1+cos(θ))/2,atol=1e-14,rtol=1e-8)
	end
	@testset "ClampedWignerdMatrix" begin
		n = 100
		
		for θ in LinRange(π/n,π-π/n,2n+1)
			d = ClampedWignerdMatrix(1,θ)
			test(d,θ)
		end

		θ = Equator()
		d = ClampedWignerdMatrix(1,θ)
		@testset "Equator" begin
		   test(d,θ) 
		end
	end
	@testset "WignerdMatrix" begin
		n = 100
		
		for θ in LinRange(π/n,π-π/n,2n+1)
			d = WignerdMatrix(1,θ)
			test(d,θ)
		end

		θ = Equator()
		d = WignerdMatrix(1,θ)
		@testset "Equator" begin
		   test(d,θ) 
		end
	end
end

@testset "d2_mn(θ)" begin
	function test(d::ClampedWignerdMatrix,θ)
		@test isapprox(d[2,1],-sin(θ)*(1+cos(θ))/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[2,0],1/2*√(3/2)*sin(θ)^2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[2,-1],-sin(θ)*(1-cos(θ))/2,atol=1e-14,rtol=1e-8)
		
		@test isapprox(d[1,1],(2cos(θ)^2+cos(θ)-1)/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[1,0],-√(3/2)*sin(θ)*cos(θ),atol=1e-14,rtol=1e-8)
		@test isapprox(d[1,-1],-(2cos(θ)^2-cos(θ)-1)/2,atol=1e-14,rtol=1e-8)

		@test isapprox(d[0,1],√(3/2)*sin(θ)*cos(θ),atol=1e-14,rtol=1e-8)
		@test isapprox(d[0,0],1/2*(3cos(θ)^2-1),atol=1e-14,rtol=1e-8)
		@test isapprox(d[0,-1],-√(3/2)*sin(θ)*cos(θ),atol=1e-14,rtol=1e-8)

		@test isapprox(d[-1,1],-(2cos(θ)^2-cos(θ)-1)/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-1,0],√(3/2)*sin(θ)*cos(θ),atol=1e-14,rtol=1e-8)
		@test isapprox(d[-1,-1],(2cos(θ)^2+cos(θ)-1)/2,atol=1e-14,rtol=1e-8)

		@test isapprox(d[-2,1],sin(θ)*(1-cos(θ))/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-2,0],1/2*√(3/2)*sin(θ)^2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-2,-1],sin(θ)*(1+cos(θ))/2,atol=1e-14,rtol=1e-8)
	end
	function test(d::WignerdMatrix,θ)
		c = ClampedWignerdMatrix(d)
		test(c,θ)

		# Extra indices
		@test isapprox(d[2,2],(1+cos(θ))^2/4,atol=1e-14,rtol=1e-8)
		@test isapprox(d[2,-2],(1-cos(θ))^2/4,atol=1e-14,rtol=1e-8)
		
		@test isapprox(d[1,2],sin(θ)*(1+cos(θ))/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[1,-2],-sin(θ)*(1-cos(θ))/2,atol=1e-14,rtol=1e-8)
		
		@test isapprox(d[0,2],1/2*√(3/2)*sin(θ)^2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[0,-2],1/2*√(3/2)*sin(θ)^2,atol=1e-14,rtol=1e-8)

		@test isapprox(d[-1,2],sin(θ)*(1-cos(θ))/2,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-1,-2],-sin(θ)*(1+cos(θ))/2,atol=1e-14,rtol=1e-8)

		@test isapprox(d[-2,2],(1-cos(θ))^2/4,atol=1e-14,rtol=1e-8)
		@test isapprox(d[-2,-2],(1+cos(θ))^2/4,atol=1e-14,rtol=1e-8)
	end

	@testset "ClampedWignerdMatrix" begin
		n = 100
		for θ in LinRange(π/n,π-π/n,2n+1)
			d = ClampedWignerdMatrix(2,θ)
			test(d,θ)
		end

		@testset "Equator" begin
			θ = Equator()
			d = ClampedWignerdMatrix(2,θ)
			test(d,θ)
		end
	end

	@testset "WignerdMatrix" begin
	    n = 100
		for θ in LinRange(π/n,π-π/n,2n+1)
			d = WignerdMatrix(2,θ)
			test(d,θ)
		end

		@testset "Equator" begin
			θ = Equator()
			d = WignerdMatrix(2,θ)
			test(d,θ)
		end
	end
end

@testset "Clebsch-Gordan" begin
	@testset "allocating" begin
		CG = WignerD.CG_j₁m₁_j₂m₂_lm(1,1,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6

		CG = WignerD.CG_j₁m₁_j₂m₂_lm(1,-1,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,-1,1,1,0,0) ≈ 1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,-1,1,1,1,0) ≈ -1/√2
		@test CG[2] ≈ WignerD.clebschgordan(1,-1,1,1,2,0) ≈ 1/√6

		CG = WignerD.CG_j₁m₁_j₂m₂_lm(1,0,1)

		@test CG[0] ≈ WignerD.clebschgordan(1,0,1,0,0,0) ≈ -1/√3
		@test CG[1] ≈ WignerD.clebschgordan(1,0,1,0,1,0) ≈ 0
		@test CG[2] ≈ WignerD.clebschgordan(1,0,1,0,2,0) ≈ √(2/3)
	end
	@testset "non-allocating" begin
		CG = zeros(0:2)

		WignerD.CG_j₁m₁_j₂m₂_lm!(1,1,1,0,CG)

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
				CG = WignerD.CG_j₁m₁_j₂m₂_lm(j1,m,j2,t)[max(abs(t),smin):smax]
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

@testset "Yⁿ₁ₘ explicit" begin
	θ,ϕ = π/2, π/4
    n = Point2D(θ,ϕ)
    Y = Ylmatrix(GSH(),1,n)

    Yexp = [1/2*cis(-ϕ)*√(3/π)*cos(θ/2)^2    1/2*cis(-ϕ)*√(3/(2π))*sin(θ)   1/2*cis(-ϕ)*√(3/π)*sin(θ/2)^2 
			-1/2*√(3/(2π))*sin(θ)               1/2*√(3/π)*cos(θ)              1/2*√(3/(2π))*sin(θ) 
			1/2*cis(ϕ)*√(3/π)*sin(θ/2)^2    -1/2*cis(ϕ)*√(3/(2π))*sin(θ)  1/2*cis(ϕ)*√(3/π)*cos(θ/2)^2]

    @test parent(collect(Y)) ≈ Yexp
end

@testset "Ylmatrix OSH and GSH" begin
	ℓ = 3
	n = Point2D(π/3,0)
	Y1 = Ylmatrix(GSH(),ℓ,n)
	Y2 = Ylmatrix(OSH(),ℓ,n)
	for m in axes(Y1,1)
		@test Y1[m,0] ≈ Y2[m]
	end
end

@testset "Ylmatrix special points" begin
    @testset "OSH" begin
        Y1 = Ylmatrix(OSH(),3,(π/2,π/2))
        Y2 = Ylmatrix(OSH(),3,(Equator(),π/2))
        @test Y1 ≈ Y2
    end
    @testset "GSH" begin
		Y1 = Ylmatrix(GSH(),3,(π/2,π/2))
        Y2 = Ylmatrix(GSH(),3,(Equator(),π/2))
        @test Y1 ≈ Y2
    end
end

@testset "Y1100 explicit" begin
	n1 = Point2D(π*rand(),2π*rand())
	n2 = Point2D(π*rand(),2π*rand())
	@test BiPoSH(OSH(),nothing,n1,n2,0,0,1,1) ≈ -√3/4π * cosχ(n1,n2)
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
		YB_00[ℓ] = BiPoSH(OSH(),nothing,n1,n2,0,0,ℓ,ℓ)
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
		YB_10_n1n2[ℓ] = BiPoSH(OSH(),nothing,n1,n2,1,0,ℓ,ℓ)
		YB_10_n2n1[ℓ] = BiPoSH(OSH(),nothing,n2,n1,1,0,ℓ,ℓ)
	end
	
	@test Yℓℓ_10 ≈ YB_10_n1n2
	@test YB_10_n1n2 ≈ -YB_10_n2n1
end

@testset "Ylmatrix conjugate" begin
	n = Point2D(π/3,π/3)
    @testset "GSH" begin
    	for j = 1:5
	        Y = Ylmatrix(GSH(),j,n)
	        for m = -j:j, n=-1:1
	        	@test Y[-m,-n] == (-1)^(m+n) * conj(Y[m,n])
	        end
	        @testset "m = 0" begin
	        	for n=-1:1
	           		@test iszero(imag(Y[0,n]))
	           	end
	        end
	    end
    end
    @testset "OSH" begin
    	j = 4
        Y = Ylmatrix(OSH(),j,n)
        for m = -j:j
        	@test Y[-m] == (-1)^m * conj(Y[m])
        end
        @testset "m = 0" begin
            @test iszero(imag(Y[0]))
        end
    end
end

@testset "Y rotation" begin
    #= VSH rotate as Y_{ℓm′}^γ (n′) = ∑_m D^ℓ_{m,m′}(α,β,γ) Y_{ℓm}^γ (n)
    Assume passive rotation by an angle β about the y axis
    in the counter-clockwise sense. This implies α = γ = 0.
    If R represents this passive rotation, we obtain n′ = R⁻¹ n
    For points on the x-z plane lying on the unit sphere 
    (equivalently lying on the prime-meridian), this would imply 
    (θ′, ϕ′=0) = (θ - β, ϕ=0).
    =#

    α, β, γ = 0, π/10, 0
    θ = π/4; θ′ = θ - β
    n = Point2D(θ,0); n′ = Point2D(θ′,0)

    for ℓ in 1:50

	    Y = Ylmatrix(GSH(),ℓ,n)
	    Y′ = Ylmatrix(GSH(),ℓ,n′)

	    D = WignerDMatrix(ℓ,α,β,γ)

	    for m′ in -ℓ:ℓ, γ in WignerD.vectorinds(ℓ)
		    @test isapprox(Y′[m′,γ],sum(D[m,m′]*Y[m,γ] for m in -ℓ:ℓ),atol=1e-13,rtol=1e-8)
		end
	end
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

	B = SHVector{ComplexF64}(LM(1:1,0:0));

	for ℓ in axes(Yℓℓ_10,1)
		
		BiPoSH!(OSH(),nothing,n1,n2,B,ℓ,ℓ,YSH1,YSH2,P,coeff,
			compute_Y₁=false,compute_Y₂=false);
		
		YB_10_n1n2[ℓ] = B[(1,0)]

		BiPoSH!(OSH(),nothing,n2,n1,B,ℓ,ℓ,YSH2,YSH1,P,coeff,
			compute_Y₁=false,compute_Y₂=false);

		YB_10_n2n1[ℓ] = B[(1,0)]
	end
	
	@test Yℓℓ_10 ≈ YB_10_n1n2
	@test YB_10_n1n2 ≈ -YB_10_n2n1
end

@testset "BiPoSH PB GSH explicit for m=0" begin
	n1 = Point2D(π/2,π/4)
	n2 = Point2D(π/2,π/2)
	Δϕ₁₂ = n1.ϕ - n2.ϕ

	j = 1

    @testset "l=0" begin
        B = BiPoSH(GSH(),PB(),n1,n2,0,0,j,j)
        B_exp = [√3/8π*(-1 + cosh(im*Δϕ₁₂))    √(3/2)/4π*sinh(im*Δϕ₁₂)     √3/8π*(1 + cosh(im*Δϕ₁₂)) 
       			-√(3/2)/4π*sinh(im*Δϕ₁₂)       -√3/4π*cosh(im*Δϕ₁₂)       -√(3/2)/4π*sinh(im*Δϕ₁₂) 
       			√3/8π*(1 + cosh(im*Δϕ₁₂))     √(3/2)/4π*sinh(im*Δϕ₁₂)      √3/8π*(-1 + cosh(im*Δϕ₁₂))]

       	@test parent(B) ≈ B_exp
   end
   @testset "l=1" begin
        B = BiPoSH(GSH(),PB(),n1,n2,1,0,j,j)
        B_exp = [3/(8*√2π)*sinh(im*Δϕ₁₂)    3/8π*cosh(im*Δϕ₁₂)     3/(8*√2π)*sinh(im*Δϕ₁₂) 
       			-3/8π*cosh(im*Δϕ₁₂)       -3/(4*√2π)*sinh(im*Δϕ₁₂)      -3/8π*cosh(im*Δϕ₁₂) 
       			3/(8*√2π)*sinh(im*Δϕ₁₂)     3/8π*cosh(im*Δϕ₁₂)      3/(8*√2π)*sinh(im*Δϕ₁₂)]

       	@test parent(B) ≈ B_exp
   end
   @testset "l=2" begin
        B = BiPoSH(GSH(),PB(),n1,n2,2,0,j,j)
        B_exp = [√(3/2)/8π*(2 + cosh(im*Δϕ₁₂))    √3/8π*sinh(im*Δϕ₁₂)       √(3/2)/8π*(-2 + cosh(im*Δϕ₁₂)) 
       			 -√3/8π*sinh(im*Δϕ₁₂)            -√(3/2)/4π*cosh(im*Δϕ₁₂)   -√3/8π*sinh(im*Δϕ₁₂) 
       			 √(3/2)/8π*(-2 + cosh(im*Δϕ₁₂))   √3/8π*sinh(im*Δϕ₁₂)       √(3/2)/8π*(2 + cosh(im*Δϕ₁₂)) ]

       	@test parent(B) ≈ B_exp
   end
end

@testset "BiPoSH OSH and PB GSH" begin
	n1 = Point2D(π/3,π/4)
	n2 = Point2D(π/3,π/3)
	s_max = 3
	SHModes = LM(0:s_max);
	for j₁ in 0:3, j₂ in 0:3
		WignerD.δ(j₁,j₂,s_max) || continue
	    B_GSH=BiPoSH(GSH(),PB(),n1,n2,SHModes,j₂,j₁)
	    B_OSH=BiPoSH(OSH(),nothing,n1,n2,SHModes,j₂,j₁)
    	@test begin
    		res = B_GSH[0,0,:] ≈ B_OSH
    		if !res
    			@show (j₁,j₂,s_max) B_GSH[0,0,:] B_OSH
    		end
    		res
    	end
    end
end

@testset "BiPoSH PB conjugate" begin
    n1 = Point2D(π/3,π/4)
	n2 = Point2D(π/3,π/3)

	@testset "OSH" begin
		@testset "m=0" begin
		    for j₁ = 1:3, j₂ = 1:3, l = abs(j₁-j₂):j₁+j₂
		    	B = BiPoSH(OSH(),nothing,n1,n2,l,0,j₁,j₂)
		    	if iseven(j₁+j₂+l)
		    		@test B == real(B)
		    	else
		    		@test B == imag(B)*im
		    	end
		    end
		end
		@testset "m and -m" begin
		    for j₁ = 1:3, j₂ = 1:3, l = abs(j₁-j₂):j₁+j₂, m=-l:l
		    	Bm = BiPoSH(OSH(),nothing,n1,n2,l,m,j₁,j₂)
		    	B₋m = BiPoSH(OSH(),nothing,n1,n2,l,-m,j₁,j₂)
		    	@test Bm ≈ (-1)^(j₁+j₂+l+m)*conj(B₋m)
		    end
		end
	end

	@testset "GSH" begin
	    @testset "m=0" begin
			function testconj(l,j₁,j₂)
			    phase = (-1)^(j₁+j₂+l)
			    b = BiPoSH(GSH(),PB(),n1,n2,l,0,j₁,j₂)
			    for α₂ = -1:1, α₁ = -1:1
			    	@test begin
			    		res = b[α₁,α₂] ≈ phase * (-1)^(α₁+α₂) * conj(b[-α₁,-α₂])
			    		if !res
			    			@show (l,j₁,j₂,α₁,α₂) b[α₁,α₂] b[-α₁,-α₂]
			    		end
			    		res
			    	end
			    end
			    # The (0,0) term is explicitly set for m=0
			    @test b[0,0] == phase * conj(b[0,0])
			end
			for j₁ = 1:4, j₂ = 1:4, l = abs(j₁ - j₂):j₁+j₂
				testconj(l,j₁,j₂)
			end
		end
		@testset "m and -m" begin
		    function testconj(l,m,j₁,j₂)
			    
			    phase = (-1)^(j₁+j₂+l+m)
			    
			    b1 = BiPoSH(GSH(),PB(),n1,n2,l,m,j₁,j₂)
			    b2 = BiPoSH(GSH(),PB(),n1,n2,l,-m,j₁,j₂)

			    for α₂ = -1:1, α₁ = -1:1
			    	@test begin
			    		res = isapprox(b1[α₁,α₂], phase * (-1)^(α₁+α₂) * conj(b2[-α₁,-α₂]), 
			    			atol=1e-14, rtol= sqrt(eps(Float64)))
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
end

@testset "BiPoSH PB ℓrange" begin
	n1 = Point2D(π/3,0)
	n2 = Point2D(π/3,π/3)
	SHModes = LM(0:5)
	ℓ_range = 1:10
	ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes)

	@testset "OSH" begin
	    B_all = BiPoSH(OSH(),nothing,n1,n2,SHModes,ℓ′ℓ)

	    for (ℓ′,ℓ) in ℓ′ℓ
	    	B = BiPoSH(OSH(),nothing,n1,n2,SHModes,ℓ′,ℓ)
	    	@test B_all[(ℓ′,ℓ)] ≈ B
	    end
	end
	@testset "GSH" begin
	    B_all = BiPoSH(GSH(),PB(),n1,n2,SHModes,ℓ′ℓ)

	    for (ℓ′,ℓ) in ℓ′ℓ
	    	B = BiPoSH(GSH(),PB(),n1,n2,SHModes,ℓ′,ℓ)
	    	@test B_all[(ℓ′,ℓ)] ≈ B
	    end
	end
end

@testset "BiPoSH PB all (l,m) one (l₁,l₂)" begin
	n1 = Point2D(π/2,0);
	n2 = Point2D(π/2,π/3);
    SHModes = LM(0:2)
    ℓ′,ℓ = 1,2
    b = BiPoSH(OSH(),nothing,n1,n2,SHModes,ℓ′,ℓ)
    for (s,t) in first(shmodes(b))
    	@test b[(s,t)] ≈ BiPoSH(OSH(),nothing,n1,n2,s,t,ℓ′,ℓ)
    end
end

@testset "BiPoSH Hansen one pair" begin

	function testPBHansen(B_PB,B_H)
		@test isapprox(B_H[0,0],B_PB[0,0],atol=1e-14,rtol=1e-10)
	    @test isapprox(B_H[1,0],B_PB[1,0] + B_PB[-1,0],atol=1e-14,rtol=1e-10)
	    @test isapprox(B_H[0,1],B_PB[0,-1] + B_PB[0,1],atol=1e-14,rtol=1e-10)
	    @test isapprox(B_H[1,1],B_PB[-1,-1] + B_PB[1,-1] + 
	    				B_PB[-1,1] + B_PB[1,1],atol=1e-14,rtol=1e-10)
	end

	@testset "real" begin
	    n1 = Point2D(π/2,0);
		n2 = Point2D(π/2,π/3);
    	j₁,j₂,l,m = 2,3,1,1
	    B_PB = BiPoSH(GSH(),PB(),n1,n2,l,m,j₁,j₂)
	    B_H = BiPoSH(GSH(),Hansen(),n1,n2,l,m,j₁,j₂)

	    testPBHansen(B_PB,B_H)
	end
	@testset "Equator" begin
		n1 = Point2D(Equator(),0);
		n2 = Point2D(Equator(),π/3);
		j₁,j₂,l,m = 2,3,2,1
		B_PB = BiPoSH(GSH(),PB(),n1,n2,l,m,j₁,j₂)
	    B_H = BiPoSH(GSH(),Hansen(),n1,n2,l,m,j₁,j₂)

	    testPBHansen(B_PB,B_H)

	    n1 = Point2D(Equator(),0);
		n2 = Point2D(Equator(),π/3);
		j₁,j₂,l,m = 2,3,2,2
		B_H = BiPoSH(GSH(),Hansen(),n1,n2,l,m,j₁,j₂)
		@test all(iszero,B_H)
	end
	@testset "NorthPole" begin
	    n1 = Point2D(NorthPole(),0);
		n2 = Point2D(NorthPole(),0);
		j₁,j₂,l,m = 2,3,2,1
		B_PB = BiPoSH(GSH(),PB(),n1,n2,l,m,j₁,j₂)
	    B_H = BiPoSH(GSH(),Hansen(),n1,n2,l,m,j₁,j₂)
	    testPBHansen(B_PB,B_H)

	    testPBHansen(B_PB,B_H)
	    n1 = Point2D(0,0);
		n2 = Point2D(0,0);
		B_PB = BiPoSH(GSH(),PB(),n1,n2,l,m,j₁,j₂)
		testPBHansen(B_PB,B_H)
	end
end

@testset "BiPoSH Hansen m=0" begin
	isimag(z) = iszero(real(z))
	function testrealimag(n1,n2)
		for j₁=1:3,j₂=1:3,l=abs(j₁-j₂):j₁+j₂
	    	B = BiPoSH(GSH(),Hansen(),n1,n2,l,0,j₁,j₂)
	    	if iseven(j₁+j₂+l)
	    		@test isreal(B[0,0])
	    		@test isimag(B[0,1])
	    		@test isimag(B[1,0])
	    		@test isreal(B[1,1])
	    	else
	    		@test isimag(B[0,0])
	    		@test isreal(B[0,1])
	    		@test isreal(B[1,0])
	    		@test isimag(B[1,1])
	    	end
	    end
	end
	@testset "both Equator" begin
		function test00(n1,n2)
			for j=1:3
				B = BiPoSH(GSH(),Hansen(),n1,n2,0,0,j,j)
				@test begin 
					res = B[0,1] ≈ conj(B[1,0])
					if !res
						@show j l B[0,1] B[1,0]
					end
					res
				end
				@test begin 
					res = B[0,1] ≈ -B[1,0]
					if !res
						@show j l B[0,1] B[1,0]
					end
					res
				end
			end
		end
	    n1,n2 = (Equator(),0),(Equator(),pi/3)
	    testrealimag(n1,n2)
	    test00(n1,n2)
	end
	@testset "both real" begin
	    n1,n2 = (pi/2,0),(pi/2,pi/3)
	    testrealimag(n1,n2)
	end
end

@testset "BiPoSH ℓrange n1n2 n2n1" begin
	
	n1 = Point2D(π/2,0);
	n2 = Point2D(π/2,π/3);

	function test_and_print_fail(arr,(ℓ′ℓind,ℓ′,ℓ),match_arr)
		@test begin
			res = false
			try
    			res = isapprox(arr[ℓ′ℓind],match_arr,atol=1e-14,rtol=1e-10)
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
		ℓ_range = 0:5;
		SHModes = LM(ℓ_range);
		ℓ′ℓ = L2L1Triangle(ℓ_range, SHModes);

		@testset "all ℓ′" begin
		    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(OSH(),nothing,n1,n2,SHModes,ℓ′ℓ)
		    Yℓ′n₁ℓn₂_all_2,Yℓ′n₂ℓn₁_all_2 = BiPoSH_n1n2_n2n1(OSH(),nothing,n1,n2,SHModes,ℓ_range)
		    @test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₂ℓn₁_all_2)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₂ℓn₁_all_2)) == ℓ′ℓ
		    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
		    	Yℓ′n₁ℓn₂ = BiPoSH(OSH(),nothing,n1,n2,SHModes,ℓ′,ℓ)
		    	Yℓ′n₂ℓn₁ = BiPoSH(OSH(),nothing,n2,n1,SHModes,ℓ′,ℓ)
		    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
		    	test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
		    	test_and_print_fail(Yℓ′n₁ℓn₂_all_2,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
		    	test_and_print_fail(Yℓ′n₂ℓn₁_all_2,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
		    end
		end

	    @testset "some ℓ′" begin
		    ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes);
		    ℓ′range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
		    if length(ℓ′range) > 1
			    ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes,ℓ′range[2:end]);
			    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(OSH(),nothing,n1,n2,SHModes,ℓ′ℓ)
			    @test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			    @test first(shmodes(Yℓ′n₂ℓn₁_all)) == ℓ′ℓ

			    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
			    	Yℓ′n₁ℓn₂ = BiPoSH(OSH(),nothing,n1,n2,SHModes,ℓ′,ℓ)
			    	Yℓ′n₂ℓn₁ = BiPoSH(OSH(),nothing,n2,n1,SHModes,ℓ′,ℓ)
			    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
		    		test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
			    end
			end
	    end
	end

	@testset "GSH PB" begin
		ℓ_range = 0:5;
		SHModes = LM(ℓ_range);
		ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes);
		
		function testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			# We test for Yʲ²ʲ¹ₗₘ_α₂α₁_n₂n₁ = (-1)^(j₁+j₂+l) * Yʲ¹ʲ²ₗₘ_α₁α₂_n₁n₂
	    	for (ℓ′,ℓ) in Iterators.product(ℓ_common,ℓ_common)
	    		Yℓ′n₁ℓn₂ = Yℓ′n₁ℓn₂_all[(ℓ′,ℓ)]
	    		Yℓn₂ℓ′n₁ = Yℓ′n₂ℓn₁_all[(ℓ,ℓ′)]
	    		for (s,t) in first(shmodes(Yℓ′n₁ℓn₂))
	    			phase = (-1)^(ℓ′ + ℓ + s)
	    			Yℓ′n₁ℓn₂_st = Yℓ′n₁ℓn₂[:,:,(s,t)]
	    			Yℓn₂ℓ′n₁_st = Yℓn₂ℓ′n₁[:,:,(s,t)]
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
	    	@test first(shmodes(Yℓ′n₁ℓn₂_all)) == first(shmodes(Yℓ′n₁ℓn₂_all_OSH))
	    	for j₂j₁ind in axes(Yℓ′n₁ℓn₂_all,1)
	    		YGSH = Yℓ′n₁ℓn₂_all[j₂j₁ind]
	    		YOSH = Yℓ′n₁ℓn₂_all_OSH[j₂j₁ind]
	    		@test first(shmodes(YGSH)) == first(shmodes(YOSH))
	    		@test YGSH[0,0,:] ≈ YOSH
	    	end
	    end

		@testset "all ℓ′" begin
		    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(GSH(),PB(),n1,n2,SHModes,ℓ′ℓ)
		    Yℓ′n₁ℓn₂_all_2,Yℓ′n₂ℓn₁_all_2 = BiPoSH_n1n2_n2n1(GSH(),PB(),n1,n2,SHModes,ℓ_range)
		    
		    @test Yℓ′n₁ℓn₂_all == Yℓ′n₁ℓn₂_all_2
		    @test Yℓ′n₂ℓn₁_all == Yℓ′n₂ℓn₁_all_2

		    @test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₂ℓn₁_all_2)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₂ℓn₁_all_2)) == ℓ′ℓ
			@testset "match with B(ℓ′,ℓ)" begin
			    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
			    	Yℓ′n₁ℓn₂ = BiPoSH(GSH(),PB(),n1,n2,SHModes,ℓ′,ℓ)
			    	Yℓ′n₂ℓn₁ = BiPoSH(GSH(),PB(),n2,n1,SHModes,ℓ′,ℓ)
			    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
			    	test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
			    end
			end
			@testset "match flip" begin
				ℓ′_range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
				ℓ_common = intersect(ℓ′_range,ℓ_range)
			    testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			end
			@testset "compare with OSH" begin
			    Yℓ′n₁ℓn₂_all_OSH,Yℓ′n₂ℓn₁_all_OSH = BiPoSH_n1n2_n2n1(OSH(),nothing,n1,n2,SHModes,ℓ′ℓ)
			    testOSH(Yℓ′n₁ℓn₂_all,Yℓ′n₁ℓn₂_all_OSH)
			    testOSH(Yℓ′n₂ℓn₁_all,Yℓ′n₂ℓn₁_all_OSH)
			end
		end

	    @testset "some ℓ′" begin
		    ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes);
		    ℓ′range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
		    if length(ℓ′range) > 1
			    ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes,ℓ′range[2:end]);
			    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(GSH(),PB(),n1,n2,SHModes,ℓ′ℓ)
			    @test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			    @test first(shmodes(Yℓ′n₂ℓn₁_all)) == ℓ′ℓ

			    @testset "match with B(ℓ′,ℓ)" begin
				    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
				    	Yℓ′n₁ℓn₂ = BiPoSH(GSH(),PB(),n1,n2,SHModes,ℓ′,ℓ)
				    	Yℓ′n₂ℓn₁ = BiPoSH(GSH(),PB(),n2,n1,SHModes,ℓ′,ℓ)
				    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
			    		test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
				    end
				end
			end
			@testset "match flip" begin
				ℓ′_range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
				ℓ_common = intersect(ℓ′_range,ℓ_range)
			    testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			end
			@testset "compare with OSH" begin
			    Yℓ′n₁ℓn₂_all_OSH,Yℓ′n₂ℓn₁_all_OSH = BiPoSH_n1n2_n2n1(OSH(),nothing,n1,n2,SHModes,ℓ′ℓ)
			    testOSH(Yℓ′n₁ℓn₂_all,Yℓ′n₁ℓn₂_all_OSH)
			    testOSH(Yℓ′n₂ℓn₁_all,Yℓ′n₂ℓn₁_all_OSH)
			end
	    end
	end

	@testset "GSH Hansen" begin
		ℓ_range = 0:5;
		SHModes = LM(ℓ_range);
		ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes);
		
		function testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			# We test for Yʲ²ʲ¹ₗₘ_α₂α₁_n₂n₁ = (-1)^(j₁+j₂+l) * Yʲ¹ʲ²ₗₘ_α₁α₂_n₁n₂
	    	for (ℓ′,ℓ) in Iterators.product(ℓ_common,ℓ_common)
	    		Yℓ′n₁ℓn₂ = Yℓ′n₁ℓn₂_all[(ℓ′,ℓ)]
	    		Yℓn₂ℓ′n₁ = Yℓ′n₂ℓn₁_all[(ℓ,ℓ′)]
	    		for (s,t) in first(shmodes(Yℓ′n₁ℓn₂))
	    			phase = (-1)^(ℓ′ + ℓ + s)
	    			Yℓ′n₁ℓn₂_st = Yℓ′n₁ℓn₂[:,:,(s,t)]
	    			Yℓn₂ℓ′n₁_st = Yℓn₂ℓ′n₁[:,:,(s,t)]
    				@test begin
    					res = isapprox(Yℓ′n₁ℓn₂_st,phase .* permutedims(Yℓn₂ℓ′n₁_st),
    								atol=1e-14,rtol=1e-10)
    					if !res
    						@show (s,t) Yℓ′n₁ℓn₂_st Yℓn₂ℓ′n₁_st
    					end
    					res
    				end
	    		end
	    	end
	    end

		@testset "all ℓ′" begin
		    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(GSH(),Hansen(),n1,n2,SHModes,ℓ′ℓ)
		    Yℓ′n₁ℓn₂_all_2,Yℓ′n₂ℓn₁_all_2 = BiPoSH_n1n2_n2n1(GSH(),Hansen(),n1,n2,SHModes,ℓ_range)
		    
		    @test Yℓ′n₁ℓn₂_all == Yℓ′n₁ℓn₂_all_2
		    @test Yℓ′n₂ℓn₁_all == Yℓ′n₂ℓn₁_all_2

		    @test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₂ℓn₁_all_2)) == ℓ′ℓ
			@test first(shmodes(Yℓ′n₂ℓn₁_all_2)) == ℓ′ℓ
			@testset "match with B(ℓ′,ℓ)" begin
			    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
			    	Yℓ′n₁ℓn₂ = BiPoSH(GSH(),Hansen(),n1,n2,SHModes,ℓ′,ℓ)
			    	Yℓ′n₂ℓn₁ = BiPoSH(GSH(),Hansen(),n2,n1,SHModes,ℓ′,ℓ)
			    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
			    	test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
			    end
			end
			@testset "match flip" begin
				ℓ′_range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
				ℓ_common = intersect(ℓ′_range,ℓ_range)
			    testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			end
		end

	    @testset "some ℓ′" begin
		    ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes);
		    ℓ′range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
		    if length(ℓ′range) > 1
			    ℓ′ℓ = L2L1Triangle(ℓ_range,SHModes,ℓ′range[2:end]);
			    Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all = BiPoSH_n1n2_n2n1(GSH(),Hansen(),n1,n2,SHModes,ℓ′ℓ)
			    @test first(shmodes(Yℓ′n₁ℓn₂_all)) == ℓ′ℓ
			    @test first(shmodes(Yℓ′n₂ℓn₁_all)) == ℓ′ℓ

			    @testset "match with B(ℓ′,ℓ)" begin
				    for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ)
				    	Yℓ′n₁ℓn₂ = BiPoSH(GSH(),Hansen(),n1,n2,SHModes,ℓ′,ℓ)
				    	Yℓ′n₂ℓn₁ = BiPoSH(GSH(),Hansen(),n2,n1,SHModes,ℓ′,ℓ)
				    	test_and_print_fail(Yℓ′n₁ℓn₂_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₁ℓn₂)
			    		test_and_print_fail(Yℓ′n₂ℓn₁_all,(ℓ′ℓind,ℓ′,ℓ),Yℓ′n₂ℓn₁)
				    end
				end
			end
			@testset "match flip" begin
				ℓ′_range = SphericalHarmonicModes.l2_range(ℓ′ℓ)
				ℓ_common = intersect(ℓ′_range,ℓ_range)
			    testflip(Yℓ′n₁ℓn₂_all,Yℓ′n₂ℓn₁_all,ℓ_common)
			end
	    end
	end
end

@testset "BiPoSH rotation" begin
    #= BiPoSh rotate as 
    B_{ℓm′}^{j₁β,j₂γ} (n₁′,n₂′) = 
    	∑_m D^ℓ_{m,m′}(α,β,γ) B_{ℓm}^^{j₁β,j₂γ} (n₁,n₂)
    Assume passive rotation by an angle β about the y axis
    in the counter-clockwise sense. This implies α = γ = 0.
    If R represents this passive rotation, we obtain n′ = R⁻¹ n
    For points on the x-z plane lying on the unit sphere 
    (equivalently lying on the prime-meridian), this would imply 
    (θ′, ϕ′=0) = (θ - β, ϕ=0).
    =#

    α, β, γ = 0, π/10, 0
    θ₁ = π/4; θ₁′ = θ₁ - β
    n₁ = Point2D(θ₁,0); n₁′ = Point2D(θ₁′,0)
    θ₂ = π/3; θ₂′ = θ₂ - β
    n₂ = Point2D(θ₂,0); n₂′ = Point2D(θ₂′,0)

    @testset "PB" begin
    	for j₁ = 1:10, j₂ = 1:10, ℓ = abs(j₁-j₂):j₁+j₂

	    	B = BiPoSH(GSH(),PB(),n₁,n₂,LM(ℓ:ℓ),j₁,j₂)
	    	B′ = BiPoSH(GSH(),PB(),n₁′,n₂′,LM(ℓ:ℓ),j₁,j₂)

		    D = WignerDMatrix(ℓ,α,β,γ)

		    for m′ in -ℓ:ℓ, γ in WignerD.vectorinds(j₂), β in WignerD.vectorinds(j₁)
			    @test isapprox(B′[β,γ,(ℓ,m′)],sum(D[m,m′]*B[β,γ,(ℓ,m)] for m in -ℓ:ℓ),atol=1e-13,rtol=1e-8)
			end
		end
    end
end

