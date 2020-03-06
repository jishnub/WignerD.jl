module WignerD

using Compat
using Reexport
using OffsetArrays
using LinearAlgebra
using Libdl
@reexport using PointsOnASphere
@reexport using SphericalHarmonics
@reexport using SphericalHarmonicArrays
@reexport using WignerSymbols
@reexport using SphericalHarmonicModes
import SphericalHarmonics: Pole

import SphericalHarmonicArrays: SHArrayOnlyFirstAxis
import SphericalHarmonicModes: ModeRange, SHModeRange

export Ylmn
export Ylmatrix
export Ylmatrix!
export djmatrix!
export djmn
export djmatrix
export BiPoSH
export BiPoSH_n1n2_n2n1
export BiPoSH!
export SphericalHarmonic
export SphericalHarmonic!
export OSH
export GSH
export Equator

struct Equator <: Real end

# Returns exp(i α π/2) = cos(α π/2) + i*sin(α π/2)
function Base.cis(α,::Equator)
	res = zero(ComplexF64)
	αmod4 = mod(Int(α),4)
	if αmod4 == 0
		res += one(res)
	elseif αmod4 == 1
		res += one(res)*im
	elseif αmod4 == 2
		res -= one(res)
	elseif αmod4 == 3
		res -= one(res)*im
	end
	return res
end

@inline Base.cis(α,::NorthPole) = one(ComplexF64)

# Returns exp(i α π) = cos(α π) + i*sin(α π) = cos(α π)
function Base.cis(α,::SouthPole)
	res = one(ComplexF64)
	if mod(Int(α),2) == 1 
		res *= -1
	end
	res
end

Base.cos(::Equator) = zero(Float64)
Base.sin(::Equator) = one(Float64)

Base.Float64(::Equator) = π/2
Base.AbstractFloat(p::Equator) = Float64(p)

Base.one(::Type{Equator}) = one(Float64)

#########################################################################
# Dictionary to cache the eigenvectors and eigenvalues of Jy 
#########################################################################

const JyEigenDict = Dict{Int,
	Tuple{OffsetVector{Float64,Vector{Float64}},OffsetArray{ComplexF64,2,Matrix{ComplexF64}}}}()

##########################################################################
# Wigner d matrix
##########################################################################

X(j,n) = sqrt((j+n)*(j-n+1))

function coeffi(j)
	N = 2j+1
	A = zeros(ComplexF64,N,N)
	coeffi!(j,A)
end

function coeffi!(j,A)

	N = 2j+1
	fill!(A,zero(eltype(A)))
	Av = reshape(@view(A[1:N^2]),N,N)
	h = Hermitian(Av)

	Av[1,1] = zero(ComplexF64)

    for i in 1:N-1
	    Av[i,i+1]=-X(j,-j+i)/2im
	end

	return h
end

function Jy_eigen!(j,A)
	
	if j in keys(JyEigenDict)
		return JyEigenDict[j]
	end
	
	A_filled = coeffi!(j,A)
	λ,v = eigen!(A_filled)
	# We know that the eigenvalues of Jy are m ∈ -j:j, 
	# so we can round λ to integers and gain accuracy
	λ = round.(λ)
	#sort the array
	if issorted(λ)
		v = OffsetArray(permutedims(v),-j:j,-j:j)
		λ = OffsetArray(λ,-j:j)
	else
		p = sortperm(λ)
		v = OffsetArray(permutedims(v[:,p]),-j:j,-j:j)
		λ = OffsetArray(λ[p],-j:j)
	end

	JyEigenDict[j] = (λ,v)

	return λ,v
end

function djmatrix_terms(θ::Real,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	dj_m_n = zero(ComplexF64)
	dj_m_n_πmθ = zero(ComplexF64)
	dj_n_m = zero(ComplexF64)

	@inbounds for μ in axes(λ,1)
		temp  = v[μ,m] * conj(v[μ,n])

		dj_m_n += cis(-λ[μ]*θ) * temp
		if m != n
			dj_n_m += cis(λ[μ]*θ) * temp
		end
		
		dj_m_n_πmθ += cis(-λ[μ]*(π-θ)) * temp
	end

	dj_m_n,dj_m_n_πmθ,dj_n_m
end

function djmatrix_terms(θ::NorthPole,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	dj_m_n = zero(ComplexF64)
	dj_m_n_πmθ = zero(ComplexF64)
	dj_n_m = zero(ComplexF64)

	@inbounds for μ in axes(λ,1)
		temp  = v[μ,m] * conj(v[μ,n])

		dj_m_n += cis(0.0) * temp
		if m != n
			dj_n_m += cis(0.0) * temp
		end
		
		dj_m_n_πmθ += cis(-λ[μ],SouthPole()) * temp
	end

	dj_m_n,dj_m_n_πmθ,dj_n_m
end

function djmatrix_terms(θ::SouthPole,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	dj_m_n = zero(ComplexF64)
	dj_m_n_πmθ = zero(ComplexF64)
	dj_n_m = zero(ComplexF64)

	@inbounds for μ in axes(λ,1)
		temp  = v[μ,m] * conj(v[μ,n])

		dj_m_n += cis(-λ[μ],θ) * temp
		if m != n
			dj_n_m += cis(λ[μ],θ) * temp
		end
		
		dj_m_n_πmθ += cis(0.0) * temp
	end

	dj_m_n,dj_m_n_πmθ,dj_n_m
end

function djmatrix_terms(θ::Equator,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	dj_m_n = zero(ComplexF64)
	dj_n_m = zero(ComplexF64)

	if !(isodd(j+m) && n == 0) && !(isodd(j+n) && m == 0)
		@inbounds for μ in axes(λ,1)
			temp  = v[μ,m] * conj(v[μ,n])

			dj_m_n += cis(-λ[μ],θ) * temp
			if m != n
				dj_n_m += cis(λ[μ],θ) * temp
			end
		end
	end

	dj_m_n,dj_m_n,dj_n_m
end

function djmatrix_fill!(dj,j,θ,m_range,n_range,λ,v,inds_covered = falses(m_range,n_range))

	m_range = intersect(m_range,-j:j)
	n_range = intersect(n_range,-j:j)

	fill!(inds_covered,false)

	for (m,n) in Iterators.product(m_range,n_range)

		# check if symmetry conditions allow the index to be evaluated
		inds_covered[m,n] && continue

		dj_m_n,dj_m_n_πmθ,dj_n_m = map(real,djmatrix_terms(θ,λ,v,m,n,j))

		dj[m,n] = dj_m_n
		inds_covered[m,n] = true
		if !iszero(m) && -m in m_range
			dj[-m,n] = dj_m_n_πmθ*(-1)^(j+n)
			inds_covered[-m,n] = true
		end

		if !iszero(n) && -n in n_range
			dj[m,-n] = dj_m_n_πmθ*(-1)^(j+m)
			inds_covered[m,-n] = true
		end

		if !(iszero(m) && iszero(n)) && -m in n_range && -n in m_range
			dj[-n,-m] = dj_m_n
			inds_covered[-n,-m] = true
		end

		if  !iszero(n) && m !=n && -n in n_range && -m in m_range
			dj[-m,-n] = dj_n_m
			inds_covered[-m,-n] = true
		end

		# transpose
		if m != n && m in n_range && n in m_range
			dj[n,m] = dj_n_m
			inds_covered[n,m] = true
		end
	end

	return dj
end

struct djindices end
struct GSHindices end
struct OSHindices end

vectorinds(j::Int) = iszero(j) ? Base.IdentityUnitRange(0:0) : Base.IdentityUnitRange(-1:1)

function get_m_n_ranges(j,::djindices;kwargs...)
	m_range = get(kwargs,:m_range,Base.IdentityUnitRange(-j:j))
	n_range = get(kwargs,:n_range,Base.IdentityUnitRange(-j:j))
	return m_range,n_range
end

function get_m_n_ranges(j,::GSHindices;kwargs...)
	m_range = get(kwargs,:m_range,Base.IdentityUnitRange(-j:j))
	n_range = get(kwargs,:n_range,vectorinds(j))
	return m_range,n_range
end

function get_m_n_ranges(j,::OSHindices;kwargs...)
	m_range = get(kwargs,:m_range,Base.IdentityUnitRange(-j:j))
	n_range = Base.IdentityUnitRange(0:0)
	return m_range,n_range
end

# Default to full range
get_m_n_ranges(j;kwargs...) = get_m_n_ranges(j,djindices();kwargs...)

function djmatrix!(dj,j,θ::Real,
	A::Matrix{ComplexF64}=zeros(ComplexF64,2j+1,2j+1),
	inds_covered = falses(axes(dj)); kwargs...)

	m_range,n_range = get_m_n_ranges(j;kwargs...)

	λ,v = Jy_eigen!(j,A)

	djmatrix_fill!(dj,j,θ,m_range,n_range,λ,v,inds_covered)
end

function djmatrix(j,θ;kwargs...)
	m_range,n_range = get_m_n_ranges(j;kwargs...)
	dj = zeros(m_range,n_range)
	A = zeros(ComplexF64,2j+1,2j+1)
	inds_covered = falses(m_range,n_range)
	djmatrix!(dj,j,θ,A,inds_covered;m_range=m_range,n_range=n_range,kwargs...)
end

djmatrix(j,x::SphericalPoint;kwargs...) = djmatrix(j,x.θ;kwargs...)
djmatrix(j,m,n,θ::Real;kwargs...) = djmatrix(j,θ,m_range=m:m,n_range=n:n;kwargs...)
djmatrix(j,m,n,x::SphericalPoint;kwargs...) = djmatrix(j,x.θ,m_range=m:m,n_range=n:n;kwargs...)

djmatrix!(dj::AbstractMatrix{<:Real},j,x::SphericalPoint,args...;kwargs...) = 
	djmatrix!(dj,j,x.θ,args...;kwargs...)
djmatrix!(dj::AbstractMatrix{<:Real},j,m,n,θ::Real,args...;kwargs...) = 
	djmatrix!(dj,j,θ,args...;m_range=m:m,n_range=n:n,kwargs...)
djmatrix!(dj::AbstractMatrix{<:Real},j,m,n,x::SphericalPoint,args...;kwargs...) = 
	djmatrix!(dj,j,x.θ,args...;m_range=m:m,n_range=n:n,kwargs...)

##########################################################################
# Generalized spherical harmonics
# Resort to spherical harmonics if n=0
##########################################################################

abstract type AbstractSH end
struct GSH <: AbstractSH end # Generalized SH
struct OSH <: AbstractSH end # Ordinary SH

# Convenience function to convert an integer to a UnitRange to be used as an array axis
const IntegerOrUnitRange = Union{Integer,AbstractUnitRange{<:Integer}}
to_unitrange(a::Integer) = Base.IdentityUnitRange(a:a)
to_unitrange(a::AbstractUnitRange{<:Integer}) = Base.IdentityUnitRange(a)

function allocate_YP(::OSH,lmax::Integer)
	YSH = SphericalHarmonics.allocate_y(lmax)
	coeff = SphericalHarmonics.compute_coefficients(lmax)
	P = SphericalHarmonics.allocate_p(lmax)
	return YSH,P,coeff
end
function compute_YP!(lmax,(θ,ϕ)::Tuple{Real,Real},Y,P,coeff,
	compute_Pl::Bool=true,compute_Y::Bool=true)

	x = cos(θ)
	compute_Pl && compute_p!(lmax,x,coeff,P)
	compute_Y && compute_y!(lmax,x,ϕ,P,Y)
end
function compute_YP!(lmax,(x,ϕ)::Tuple{Pole,Real},Y,P,coeff,
	compute_Pl::Bool=true,compute_Y::Bool=true)

	compute_Pl && compute_p!(lmax,x,coeff,P)
	compute_Y && compute_y!(lmax,x,ϕ,P,Y)
end

function compute_YP!(lmax,(x,ϕ)::Tuple{Equator,Real},Y,P,coeff,
	compute_Pl::Bool=true,compute_Y::Bool=true)

	compute_Pl && compute_p!(lmax,0,coeff,P)
	compute_Y && compute_y!(lmax,ϕ,P,Y)
end

function Ylmatrix(::GSH,l::Integer,(θ,ϕ)::Tuple{Real,Real};kwargs...)

	m_range,n_range = get_m_n_ranges(l,GSHindices();kwargs...)

	dj_θ = djmatrix(l,θ;kwargs...,m_range=m_range,n_range=n_range)
	Y = zeros(ComplexF64,axes(dj_θ)...)
	Ylmatrix!(GSH(),Y,dj_θ,l,(θ,ϕ);n_range=n_range,kwargs...,compute_d_matrix=false)
end

function check_indices(arr,indranges)
	for (dim,indrange) in enumerate(indranges)
		all(map(x->x in axes(arr,dim),extrema(indrange))) || throw(BoundsError(arr,indranges))
	end
end

function Ylmatrix(::GSH,dj_θ::AbstractMatrix{<:Real},l::Integer,
	(θ,ϕ)::Tuple{Real,Real};kwargs...)

	m_range,n_range = get_m_n_ranges(l,GSHindices();kwargs...)
	
	check_indices(dj_θ,(m_range,n_range))

	Y = zeros(ComplexF64,m_range,n_range)
	Ylmatrix!(GSH(),Y,dj_θ,l,(θ,ϕ);compute_d_matrix=false,
		m_range=m_range,n_range=n_range,kwargs...)
end

function Ylmatrix(::OSH,l::Integer,(θ,ϕ)::Tuple{Real,Real};kwargs...)
	YSH,P,coeff = allocate_YP(OSH(),l)
	Ylmatrix!(OSH(),YSH,l,(θ,ϕ),P,coeff;kwargs...)
end

function Ylmatrix!(::GSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},
	l::Integer,(θ,ϕ)::Tuple{Real,Real},args...;kwargs...)

	m_range,n_range = get_m_n_ranges(l,GSHindices();kwargs...)

	check_indices(dj_θ,(m_range,n_range))
	check_indices(Y,(m_range,n_range))

	if get(kwargs,:compute_d_matrix,true)
		djmatrix!(dj_θ,l,θ,args...;kwargs...,m_range=m_range,n_range=n_range)
	end

	@inbounds for n in n_range,m in m_range
		Y[m,n] = √((2l+1)/4π) * dj_θ[m,n] * cis(m*ϕ)
	end
	return Y
end

function Ylmatrix!(::OSH,YSH::AbstractVector{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{Real,Real},
	Plm_cosθ::AbstractVector{<:Real},Pcoeff;kwargs...)

	m_range = get_m_n_ranges(l,OSHindices();kwargs...) |> first

	compute_Pl = get(kwargs,:compute_Pl,true)
	compute_Ylm = get(kwargs,:compute_Ylm,true)
	compute_YP!(l,(θ,ϕ),YSH,Plm_cosθ,Pcoeff,compute_Pl,compute_Ylm)

	OffsetVector(YSH[index_y(l,m_range)],m_range)
end

Ylmatrix(::GSH,l::Integer,m::IntegerOrUnitRange,n::IntegerOrUnitRange,
	(θ,ϕ)::Tuple{Real,Real};kwargs...) = 
	Ylmatrix(GSH(),l,(θ,ϕ);kwargs...,
		m_range=to_unitrange(m),n_range=to_unitrange(n))

Ylmatrix(::OSH,l::Integer,m::IntegerOrUnitRange,
	(θ,ϕ)::Tuple{Real,Real};kwargs...) = 
	Ylmatrix(OSH(),l,(θ,ϕ);kwargs...,m_range=to_unitrange(m))

Ylmatrix(::GSH,l::Integer,m::IntegerOrUnitRange,n::IntegerOrUnitRange,
	x::SphericalPoint;kwargs...) = 
	Ylmatrix(GSH(),l,(x.θ,x.ϕ);kwargs...,
		m_range=to_unitrange(m),n_range=to_unitrange(n))

Ylmatrix(::OSH,l::Integer,m::IntegerOrUnitRange,
	x::SphericalPoint;kwargs...) = 
	Ylmatrix(OSH(),l,(x.θ,x.ϕ);kwargs...,
		m_range=to_unitrange(m))

Ylmatrix(T::AbstractSH,l::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,l,(x.θ,x.ϕ);kwargs...)

Ylmatrix( ::GSH,dj_θ::AbstractMatrix{<:Real},l::Integer,
	m::IntegerOrUnitRange,n::IntegerOrUnitRange,
	(θ,ϕ)::Tuple{Real,Real};kwargs...) = 
	Ylmatrix(GSH(),dj_θ,l,(θ,ϕ);kwargs...,
		m_range=to_unitrange(m),n_range=to_unitrange(n))

Ylmatrix( ::GSH,dj_θ::AbstractMatrix{<:Real},l::Integer,
	m::IntegerOrUnitRange,n::IntegerOrUnitRange,
	x::SphericalPoint;kwargs...) = 
	Ylmatrix(GSH(),dj_θ,l,(x.θ,x.ϕ);kwargs...,
		m_range=to_unitrange(m),n_range=to_unitrange(n))

Ylmatrix( T::AbstractSH,dj_θ::AbstractMatrix{<:Real},l::Integer,
	x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,dj_θ,l,(x.θ,x.ϕ);kwargs...)

##########################################################################
# Spherical harmonics
##########################################################################

function SphericalHarmonic(args...;kwargs...)
	Ylmatrix(OSH(),args...;kwargs...)
end

function SphericalHarmonic!(Y::AbstractVector{<:Complex},args...;kwargs...)
	Ylmatrix!(OSH(),Y,args...;kwargs...)
end

##########################################################################
# Bipolar Spherical harmonics
##########################################################################

##################################################################################################


# BiPoSH Yℓ₁ℓ₂LM(n₁,n₂)
# methods for ordinary and generalized spherical harmonics
function allocate_Y₁Y₂(::OSH,ℓ′ℓ_smax::L₂L₁Δ;kwargs...)
	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))
	ll′max = max(lmax,l′max)
	allocate_Y₁Y₂(OSH(),ll′max;kwargs...)
end
function allocate_Y₁Y₂(::OSH,lmax::Integer;kwargs...)
	YSH_n₁ = SphericalHarmonics.allocate_y(lmax)
	YSH_n₂ = SphericalHarmonics.allocate_y(lmax)
	coeff = SphericalHarmonics.compute_coefficients(lmax)
	P = SphericalHarmonics.allocate_p(lmax)
	return YSH_n₁,YSH_n₂,P,coeff
end
function allocate_Y₁Y₂(::GSH,ℓ₁,ℓ₂;β=vectorinds(ℓ₁),γ=vectorinds(ℓ₂),kwargs...)
	Yℓ₁n₁ = zeros(ComplexF64,-ℓ₁:ℓ₁,β)
	Yℓ₂n₂ = zeros(ComplexF64,-ℓ₂:ℓ₂,γ)
	dℓ₁n₁ = zeros(axes(Yℓ₁n₁))
	dℓ₂n₂ = zeros(axes(Yℓ₂n₂))
	return Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,dℓ₂n₂
end
@inline function allocate_Y₁Y₂(::GSH,lmax::Integer;kwargs...)
	allocate_Y₁Y₂(GSH(),lmax,lmax;kwargs...)
end
function allocate_Y₁Y₂(::GSH,ℓ′ℓ_smax::L₂L₁Δ;kwargs...)
	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))
	ll′max = max(lmax,l′max)
	allocate_Y₁Y₂(GSH(),ll′max;kwargs...)
end

function SHModes_slice(SHModes::SHM,ℓ′,ℓ) where {SHM<:SHModeRange}
	l_SHModes = l_range(SHModes)
	m_SHModes = m_range(SHModes)
	l_range_ℓ′ℓ = intersect(abs(ℓ-ℓ′):ℓ+ℓ′,l_SHModes)
	m_range_ℓ′ℓ = -maximum(l_range_ℓ′ℓ):maximum(l_range_ℓ′ℓ)
	m_range_ℓ′ℓ = intersect(m_range_ℓ′ℓ,m_SHModes)
	SHM(l_range_ℓ′ℓ,m_range_ℓ′ℓ)
end

function allocate_BSH(::OSH,ℓ′ℓ_smax::L₂L₁Δ,SHModes::SHM;kwargs...) where {SHM<:SHModeRange}
	T = SHVector{ComplexF64,Vector{ComplexF64},Tuple{SHM}}
	Yℓ′ℓ = SHVector{T}(undef,ℓ′ℓ_smax)
	for (ind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)
		modes_section = SHModes_slice(SHModes,ℓ′,ℓ)
		Yℓ′ℓ[ind] = T(zeros(ComplexF64,length(modes_section)),
						(modes_section,),(1,))
	end
	return Yℓ′ℓ
end

function allocate_BSH(::GSH,ℓ′ℓ_smax::L₂L₁Δ,SHModes::SHM;kwargs...) where {SHM<:SHModeRange}

	ℓ′,ℓ = first(ℓ′ℓ_smax)

	# Need to evaluate all components to be able to swap them in the recursion
	β = Base.IdentityUnitRange(-1:1)
	γ = Base.IdentityUnitRange(-1:1)

	T = SHArray{ComplexF64,3,OffsetArray{ComplexF64,3,Array{ComplexF64,3}},
			Tuple{SHM,Base.IdentityUnitRange,Base.IdentityUnitRange},1}

	Yℓ′ℓ = SHVector{T}(undef,ℓ′ℓ_smax)	

	for (ind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)

		modes_section = SHModes_slice(SHModes,ℓ′,ℓ)
		Yℓ′ℓ[ind] = T(zeros(ComplexF64,length(modes_section),β,γ),
			(modes_section,β,γ),(1,))
	end
	return Yℓ′ℓ
end

function allocate_BSH(ASH::AbstractSH,ℓ_range::AbstractUnitRange,
	SHModes::SHModeRange;kwargs...)

	ℓ′ℓ_smax = L₂L₁Δ(ℓ_range,SHModes)
	allocate_BSH(ASH,ℓ′ℓ_smax,SHModes)
end

function BiPoSH(::GSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	s::Integer,t::Integer,ℓ₁::Integer,ℓ₂::Integer;
	β::Union{AbstractUnitRange,Integer}=vectorinds(ℓ₁),
	γ::Union{AbstractUnitRange,Integer}=vectorinds(ℓ₂))
	
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied for ℓ₁=$ℓ₁, ℓ₂=$ℓ₂ and s=$s")
	@assert(abs(t)<=s,"abs(t)<=s not satisfied for t=$t and s=$s")

	β,γ = map(to_unitrange,(β,γ))

	temp_arrs = allocate_Y₁Y₂(GSH(),ℓ₁,ℓ₂,β=β,γ=γ)

	Yℓ₁ℓ₂n₁n₂ = SHArray(LM(s:s,t:t),Base.IdentityUnitRange(-1:1),Base.IdentityUnitRange(-1:1))

	BiPoSH!(GSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ₁ℓ₂n₁n₂,ℓ₁,ℓ₂,
		temp_arrs...;β=β,γ=γ,
		compute_dℓ₁=true,compute_dℓ₂=true,
		compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true)

	Yℓ₁ℓ₂n₁n₂[1,:,:]
end

function BiPoSH(::OSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	s::Integer,t::Integer,ℓ₁::Integer,ℓ₂::Integer)
	
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied for ℓ₁=$ℓ₁, ℓ₂=$ℓ₂ and s=$s")

	Yℓ₁ℓ₂n₁n₂ = SHVector(LM(s:s,t:t))

	BiPoSH!(OSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ₁ℓ₂n₁n₂,ℓ₁,ℓ₂,
		allocate_Y₁Y₂(OSH(),max(ℓ₁,ℓ₂))...,compute_Y₁=true,compute_Y₂=true)

	Yℓ₁ℓ₂n₁n₂[1]
end

function BiPoSH(::GSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,ℓ₁::Integer,ℓ₂::Integer,args...;
	β::Union{Integer,AbstractUnitRange} = vectorinds(ℓ₁),
	γ::Union{Integer,AbstractUnitRange} = vectorinds(ℓ₂),
	kwargs...)

	β,γ = map(to_unitrange,(β,γ))

	modes_section = SHModes_slice(SHModes,ℓ₁,ℓ₂)
	B = SHArray(modes_section,Base.IdentityUnitRange(-1:1),Base.IdentityUnitRange(-1:1))

	temp_arrs = allocate_Y₁Y₂(GSH(),ℓ₁,ℓ₂,β=β,γ=γ)

	BiPoSH!(GSH(),x1,x2,B,ℓ₁,ℓ₂,
		temp_arrs...,args...;
		kwargs...,β=β,γ=γ,compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true)
end

function BiPoSH(::OSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,ℓ₁::Integer,ℓ₂::Integer,
	args...;kwargs...)

	modes_section = SHModes_slice(SHModes,ℓ₁,ℓ₂)
	B = SHVector(modes_section)

	BiPoSH!(OSH(),x1,x2,B,ℓ₁,ℓ₂,
		allocate_Y₁Y₂(OSH(),max(ℓ₁,ℓ₂))...,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

function BiPoSH(ASH::AbstractSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHM,ℓ′ℓ_smax::L₂L₁Δ,
	args...;kwargs...) where {SHM<:SHModeRange}

	Yℓ′n₁ℓn₂ = allocate_BSH(ASH,ℓ′ℓ_smax,SHModes;kwargs...)

	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))

	BiPoSH!(ASH,x1,x2,Yℓ′n₁ℓn₂,ℓ′ℓ_smax,
		allocate_Y₁Y₂(ASH,max(l′max,lmax);kwargs...)...,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

BiPoSH(ASH::AbstractSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,ℓ_range::AbstractUnitRange,args...;kwargs...) = 
	BiPoSH(ASH,SHModes,L₂L₁Δ(ℓ_range,SHModes),args...;kwargs...)

BiPoSH(ASH::AbstractSH,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH(ASH,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

maybeslice(::OSH,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁;kwargs...) = (Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁)

function maybeslice(::GSH,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁;kwargs...)
	β = get(kwargs,:β,Base.IdentityUnitRange(-1:1))
	γ = get(kwargs,:γ,Base.IdentityUnitRange(-1:1))

	if (β != -1:1) || (γ != -1:1)
		Yℓ′n₁ℓn₂ = SHVector([SHArray(sa[:,β,γ],(SHModes,),(1,)) for sa in Yℓ′n₁ℓn₂],ℓ′ℓ_smax)
		Yℓ′n₂ℓn₁ = SHVector([SHArray(sa[:,β,γ],(SHModes,),(1,)) for sa in Yℓ′n₂ℓn₁],ℓ′ℓ_smax)
	end

	Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁
end

function BiPoSH_n1n2_n2n1(ASH::AbstractSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHM,ℓ′ℓ_smax::L₂L₁Δ,args...;kwargs...) where {SHM<:SHModeRange}

	Yℓ′n₁ℓn₂ = allocate_BSH(ASH,ℓ′ℓ_smax,SHModes;kwargs...)
	Yℓ′n₂ℓn₁ = allocate_BSH(ASH,ℓ′ℓ_smax,SHModes;kwargs...)

	Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,dℓ₂n₂ = allocate_Y₁Y₂(ASH,ℓ′ℓ_smax;kwargs...,β=-1:1,γ=-1:1)

	BiPoSH!(ASH,x1,x2,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,dℓ₂n₂,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true,β=-1:1,γ=-1:1)

	# Slicing, if any, happens after the calculation is done
	Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁ = maybeslice(ASH,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁;kwargs...)
end

BiPoSH_n1n2_n2n1(ASH::AbstractSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHM,ℓ_range::AbstractUnitRange,args...;kwargs...) where {SHM<:SHModeRange} = 
	BiPoSH_n1n2_n2n1(ASH,x1,x2,SHModes,L₂L₁Δ(ℓ_range,SHModes),args...;kwargs...)

BiPoSH_n1n2_n2n1(ASH::AbstractSH,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH_n1n2_n2n1(ASH,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

# For one (ℓ,ℓ′) just unpack the x1,x2 into (θ,ϕ) and and call BiPoSH_compute!
# These methods compute the monopolar YSH and pass them on to BiPoSH_compute!
function BiPoSH!(::OSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	B::AbstractVector,
	SHModes::SHModeRange,
	ℓ₁::Integer,ℓ₂::Integer,
	YSH_n₁::AbstractVector{<:Complex},
	YSH_n₂::AbstractVector{<:Complex},
	P::AbstractVector{<:Real},coeff;
	compute_Y₁=true,compute_Y₂=true,
	CG = zeros(0:ℓ₁ + ℓ₂),
	w3j = zeros(ℓ₁ + ℓ₂ + 1),
	kwargs...)

	compute_Y₁ && compute_YP!(ℓ₁,(θ₁,ϕ₁),YSH_n₁,P,coeff)
	compute_Y₂ && compute_YP!(ℓ₂,(θ₂,ϕ₂),YSH_n₂,P,coeff)

	lib = nothing
	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do 
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		Yℓ₁n₁ = OffsetVector(@view(YSH_n₁[index_y(ℓ₁)]),-ℓ₁:ℓ₁)
		Yℓ₂n₂ = OffsetVector(@view(YSH_n₂[index_y(ℓ₂)]),-ℓ₂:ℓ₂)
		
		BiPoSH_compute!(OSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),B,SHModes,ℓ₁,ℓ₂,
			Yℓ₁n₁,Yℓ₂n₂,wig3j_fn_ptr;kwargs...,w3j=w3j,CG=CG)
	finally
		Libdl.dlclose(lib)
	end

	return B
end

@inline BiPoSH!(::OSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	B::SHVector,ℓ₁::Integer,args...;kwargs...) = 
	BiPoSH!(OSH(),x1,x2,B,shmodes(B),ℓ₁,args...;kwargs...)

function BiPoSH!(::GSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	B::AbstractArray{<:Complex,3},
	SHModes::SHModeRange,
	ℓ₁::Integer,ℓ₂::Integer,
	Yℓ₁n₁::AbstractMatrix{<:Complex},
	Yℓ₂n₂::AbstractMatrix{<:Complex},
	dℓ₁n₁::AbstractMatrix{<:Real},
	dℓ₂n₂::AbstractMatrix{<:Real},
	A_djcoeffi = zeros(ComplexF64,2max(ℓ₁,ℓ₂)+1,2max(ℓ₁,ℓ₂)+1),
	dj_inds_flag = falses(-(max(ℓ₁,ℓ₂)+1):(max(ℓ₁,ℓ₂)+1),-(max(ℓ₁,ℓ₂)+1):(max(ℓ₁,ℓ₂)+1));
	compute_Y₁=true,compute_Y₂=true,
	β::Union{AbstractUnitRange,Integer}=vectorinds(ℓ₁),
	γ::Union{AbstractUnitRange,Integer}=vectorinds(ℓ₂),
	w3j = zeros(ℓ₁+ℓ₂+1),CG = zeros(0:ℓ₁+ℓ₂),kwargs...)

	β = intersect(to_unitrange(β),vectorinds(ℓ₁))
	γ = intersect(to_unitrange(γ),vectorinds(ℓ₂))

	compute_Y₁ && Ylmatrix!(GSH(),Yℓ₁n₁,dℓ₁n₁,ℓ₁,(θ₁,ϕ₁),A_djcoeffi,dj_inds_flag,n_range=β)
	compute_Y₂ && Ylmatrix!(GSH(),Yℓ₂n₂,dℓ₂n₂,ℓ₂,(θ₂,ϕ₂),A_djcoeffi,dj_inds_flag,n_range=γ,
					# don't recompute dj if unnecessary
					compute_d_matrix = !( compute_Y₁ && (ℓ₁ == ℓ₂) && (θ₁ == θ₂) && (dℓ₁n₁ === dℓ₂n₂) ) )

	lib = nothing
	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do 
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		BiPoSH_compute!(GSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),B,SHModes,ℓ₁,ℓ₂,
			Yℓ₁n₁,Yℓ₂n₂,wig3j_fn_ptr;kwargs...,w3j=w3j,CG=CG,β=β,γ=γ)
	finally
		Libdl.dlclose(lib)
	end

	return B
end

@inline BiPoSH!(::GSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	B::SHArrayOnlyFirstAxis,ℓ₁::Integer,args...;kwargs...) = 
	BiPoSH!(GSH(),x1,x2,B,shmodes(B),ℓ₁,args...;kwargs...)

@inline BiPoSH!(ASH::AbstractSH,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH!(ASH,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

"""
	BiPoSH!(OSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂::AbstractVector{<:SHVector},
	SHModes::SHModeRange,ℓ′ℓ_smax::L₂L₁Δ,args...;kwargs...)
	Compute BiPoSH for a range in ℓ and ℓ′
"""
function BiPoSH!(::OSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHVector},
	ℓ′ℓ_smax::L₂L₁Δ,
	YSH_n₁::AbstractVector{<:Complex},
	YSH_n₂::AbstractVector{<:Complex},
	P::AbstractVector{<:Real},coeff;
	CG = zeros( 0:(maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax))) ),
	w3j = zeros( maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax)) + 1),
	wig3j_fn_ptr=nothing,
	compute_Y₁=true,compute_Y₂=true,kwargs...)

	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))

	compute_Y₁ && compute_YP!(l′max,(θ₁,ϕ₁),YSH_n₁,P,coeff)
	compute_Y₂ && compute_YP!(lmax,(θ₂,ϕ₂),YSH_n₂,P,coeff)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)

			BiPoSH!(OSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂[ℓ′ℓind],
				shmodes(Yℓ′n₁ℓn₂[ℓ′ℓind]),ℓ′,ℓ,
				YSH_n₁,YSH_n₂,P,coeff;
				CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
				compute_Y₁=!compute_Y₁,compute_Y₂=!compute_Y₂)
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂
end

function BiPoSH!(::GSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHArray{<:Number,3}},
	ℓ′ℓ_smax::L₂L₁Δ,
	Yℓ₁n₁::AbstractMatrix{<:Complex},
	Yℓ₂n₂::AbstractMatrix{<:Complex},
	dℓ₁n₁::AbstractMatrix{<:Real},
	dℓ₂n₂::AbstractMatrix{<:Real};
	CG = zeros( 0:(maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax))) ),
	w3j = zeros( maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax)) + 1),
	wig3j_fn_ptr=nothing,
	compute_Y₁=true,compute_Y₂=true,kwargs...)

	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))
	l′lmax = max(lmax,l′max)

	A_djcoeffi = zeros(ComplexF64,2l′lmax+1,2l′lmax+1)
	dj_inds_flag = falses(-l′lmax:l′lmax,-l′lmax:l′lmax)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)

			BiPoSH!(GSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂[ℓ′ℓind],
				shmodes(Yℓ′n₁ℓn₂[ℓ′ℓind]),ℓ′,ℓ,
				Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,dℓ₂n₂,A_djcoeffi,dj_inds_flag;
				CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
				compute_Y₁=compute_Y₁,compute_Y₂=compute_Y₂)
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂
end

@inline BiPoSH!(::OSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHVector},
	Y1::AbstractVector{<:Complex},args...;kwargs...) = 
	BiPoSH!(OSH(),x1,x2,Yℓ′n₁ℓn₂,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

@inline BiPoSH!(::GSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHArrayOnlyFirstAxis},
	Y1::AbstractMatrix{<:Complex},args...;kwargs...) = 
	BiPoSH!(GSH(),x1,x2,Yℓ′n₁ℓn₂,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

"""
	BiPoSH!(OSH(),Yℓ′n₁ℓn₂::AbstractVector{<:SHVector},
	Yℓ′n₂ℓn₁::AbstractVector{<:SHVector},
	SHModes::SHModeRange,ℓ′ℓ_smax::L₂L₁Δ,args...;kwargs...)
	Compute BiPoSH for a range in ℓ and ℓ′ by switching the two points
	Returns Yℓ′n₁ℓn₂ and Yℓ′n₂ℓn₁
"""
function BiPoSH!(::OSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHVector{<:Number}},
	Yℓ′n₂ℓn₁::AbstractVector{<:SHVector{<:Number}},
	ℓ′ℓ_smax::L₂L₁Δ,
	YSH_n₁::AbstractVector{<:Number},
	YSH_n₂::AbstractVector{<:Number},
	P::AbstractVector{<:Real},coeff;
	CG = zeros( 0:(maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax))) ),
	w3j = zeros( maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax)) + 1),
	compute_Y₁=true,
	compute_Y₂=true,kwargs...)

	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))
	ll′max = max(lmax,l′max)

	compute_Y₁ && compute_YP!(ll′max,(θ₁,ϕ₁),YSH_n₁,P,coeff)
	compute_Y₂ && compute_YP!(ll′max,(θ₂,ϕ₂),YSH_n₂,P,coeff)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (indℓ′ℓ,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)

			# In one pass we can compute Yℓ′n₁ℓn₂ and Yℓn₂ℓ′n₁

			Yℓ′n₁ℓn₂_st = Yℓ′n₁ℓn₂[indℓ′ℓ]
			Yℓ′n₂ℓn₁_st = Yℓ′n₂ℓn₁[indℓ′ℓ]

			# We use Yℓ′n₂ℓn₁ = (-1)^(ℓ+ℓ′+s) Yℓn₁ℓ′n₂
			# and Yℓ′n₁ℓn₂ = (-1)^(ℓ+ℓ′+s) Yℓn₂ℓ′n₁
			# Precomputation of the RHS would have happened if ℓ′<ℓ, 
			# as the modes are sorted in order of increasing ℓ

			if (ℓ,ℓ′) in ℓ′ℓ_smax && ℓ′<ℓ
				# In this case Yℓn₁ℓ′n₂ and Yℓn₂ℓ′n₁ have already been computed
				# This means we can evaluate Yℓ′n₂ℓn₁ and Yℓ′n₁ℓn₂ using the formulae
				# presented above

				indℓℓ′ = modeindex(ℓ′ℓ_smax,(ℓ,ℓ′))
				Yℓn₁ℓ′n₂_st = Yℓ′n₁ℓn₂[indℓℓ′]
				Yℓn₂ℓ′n₁_st = Yℓ′n₂ℓn₁[indℓℓ′]

				for (indst,(s,t)) in enumerate(shmodes(Yℓ′n₂ℓn₁_st))
					Yℓ′n₂ℓn₁_st[indst] = (-1)^(ℓ+ℓ′+s)*Yℓn₁ℓ′n₂_st[indst]
					Yℓ′n₁ℓn₂_st[indst] = (-1)^(ℓ+ℓ′+s)*Yℓn₂ℓ′n₁_st[indst]
				end
			else
				# Default case, where we need to evaluate both

				BiPoSH!(OSH(),(θ₂,ϕ₂),(θ₁,ϕ₁),Yℓ′n₂ℓn₁_st,ℓ′,ℓ,
					YSH_n₂,YSH_n₁,P,coeff;
					CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=!compute_Y₁,compute_Y₂=!compute_Y₂,kwargs...)

				BiPoSH!(OSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂_st,ℓ′,ℓ,
					YSH_n₁,YSH_n₂,P,coeff;
					CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=!compute_Y₁,compute_Y₂=!compute_Y₂,kwargs...)
			end
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁
end

function BiPoSH!(::GSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHArray{<:Number,3}},
	Yℓ′n₂ℓn₁::AbstractVector{<:SHArray{<:Number,3}},
	ℓ′ℓ_smax::L₂L₁Δ,
	Yℓ₁n₁::AbstractMatrix{<:Complex},
	Yℓ₂n₂::AbstractMatrix{<:Complex},
	dℓ₁n₁::AbstractMatrix{<:Real},
	dℓ₂n₂::AbstractMatrix{<:Real};
	CG = zeros( 0:(maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax))) ),
	w3j = zeros( maximum(l₁_range(ℓ′ℓ_smax)) + maximum(l₂_range(ℓ′ℓ_smax)) + 1),
	compute_Y₁=true,compute_Y₂=true,kwargs...)

	lmax = maximum(l₁_range(ℓ′ℓ_smax))
	l′max = maximum(l₂_range(ℓ′ℓ_smax))
	l′lmax = max(lmax,l′max)

	A_djcoeffi = zeros(ComplexF64,2l′lmax+1,2l′lmax+1)
	dj_inds_flag = falses(-l′lmax:l′lmax,-l′lmax:l′lmax)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (ind_j₂j₁,(j₂,j₁)) in enumerate(ℓ′ℓ_smax)

			# In one pass we can compute Yℓ′n₁ℓn₂ and Yℓn₂ℓ′n₁

			Pʲ²ʲ¹ₗₘ_n₁n₂ = Yℓ′n₁ℓn₂[ind_j₂j₁]
			Pʲ²ʲ¹ₗₘ_n₂n₁ = Yℓ′n₂ℓn₁[ind_j₂j₁]

			lm_iter = shmodes(Pʲ²ʲ¹ₗₘ_n₁n₂)

			# We use the relation between the helicity basis components 
			# Pʲ²ʲ¹ₗₘ_α₂α₁(n₁,n₂) = (-1)ʲ¹⁺ʲ²⁺ˡ Pʲ¹ʲ²ₗₘ_α₁α₂(n₂,n₁)
			# and Pʲ²ʲ¹ₗₘ_α₂α₁(n₂,n₁) = (-1)ʲ¹⁺ʲ²⁺ˡ Pʲ¹ʲ²ₗₘ_α₁α₂(n₁,n₂)
			# Precomputation of the RHS would have happened if j₂ < j₁, 
			# as the modes are sorted in order of increasing j₁

			if (j₁,j₂) in ℓ′ℓ_smax && j₂ < j₁
				# In this case Pʲ¹ʲ²ₗₘ_α₁α₂(n₁,n₂) and Pʲ¹ʲ²ₗₘ_α₁α₂(n₂,n₁) have already been computed
				# This means we may evaluate Pʲ²ʲ¹ₗₘ_α₂α₁(n₂,n₁) and Pʲ²ʲ¹ₗₘ_α₂α₁(n₁,n₂) using the formulae
				# presented above

				ind_j₁j₂ = modeindex(ℓ′ℓ_smax,(j₁,j₂))

				Pʲ¹ʲ²ₗₘ_n₁n₂ = Yℓ′n₁ℓn₂[ind_j₁j₂]
				Pʲ¹ʲ²ₗₘ_n₂n₁ = Yℓ′n₂ℓn₁[ind_j₁j₂]

				for α₂ in vectorinds(j₂), α₁ in vectorinds(j₁)
					for (ind_lm,(l,m)) in enumerate(lm_iter)
						phase = (-1)^(j₁+j₂+l)
						Pʲ²ʲ¹ₗₘ_n₂n₁[ind_lm,α₂,α₁] = phase * Pʲ¹ʲ²ₗₘ_n₁n₂[ind_lm,α₁,α₂]
						Pʲ²ʲ¹ₗₘ_n₁n₂[ind_lm,α₂,α₁] = phase * Pʲ¹ʲ²ₗₘ_n₂n₁[ind_lm,α₁,α₂]
					end
				end
			else
				# Default case, where we need to evaluate both

				d2 = (j₁ == j₂) && (θ₁ == θ₂) ? dℓ₁n₁ : dℓ₂n₂

				BiPoSH!(GSH(),(θ₂,ϕ₂),(θ₁,ϕ₁),Pʲ²ʲ¹ₗₘ_n₂n₁,j₂,j₁,
					Yℓ₂n₂,Yℓ₁n₁,d2,dℓ₁n₁,A_djcoeffi,dj_inds_flag;
					kwargs...,CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=true,compute_Y₂=true)

				BiPoSH!(GSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Pʲ²ʲ¹ₗₘ_n₁n₂,j₂,j₁,
					Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,d2,A_djcoeffi,dj_inds_flag;
					kwargs...,CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=(j₁ != j₂),compute_Y₂=(j₁ != j₂))
			end
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁
end

@inline BiPoSH!(::OSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHVector},Yℓ′n₂ℓn₁::SHVector{<:SHVector},
	Y1::AbstractVector{<:Complex},args...;kwargs...) = 
	BiPoSH!(OSH(),x1,x2,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

@inline BiPoSH!(::GSH,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHArrayOnlyFirstAxis},
	Yℓ′n₂ℓn₁::SHVector{<:SHArrayOnlyFirstAxis},
	Y1::AbstractMatrix{<:Complex},args...;kwargs...) = 
	BiPoSH!(GSH(),x1,x2,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

# The actual functions that do the calculation for one pair of (ℓ₁,ℓ₂) and 
# (θ₁,ϕ₁) and (θ₂,ϕ₂). The BiPoSH! functions call these.
# We assume that the monopolar YSH are already computed in the BiPoSH! calls
function BiPoSH_compute!(::GSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ₁ℓ₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,ℓ₁::Integer,ℓ₂::Integer,
	Yℓ₁n₁::AbstractMatrix{<:Number},Yℓ₂n₂::AbstractMatrix{<:Number},wig3j_fn_ptr;
	β::AbstractUnitRange=vectorinds(ℓ₁),γ::AbstractUnitRange=vectorinds(ℓ₂),
	w3j = zeros(ℓ₁+ℓ₂+1),CG = zeros(0:ℓ₁+ℓ₂),
	kwargs...)

	fill!(Yℓ₁ℓ₂n₁n₂,zero(eltype(Yℓ₁ℓ₂n₁n₂)))

	lm_modes_ℓ₁ℓ₂ = SHModes_slice(lm_modes,ℓ₁,ℓ₂)
	l_valid = l_range(lm_modes_ℓ₁ℓ₂)
	m_valid = m_range(lm_modes_ℓ₁ℓ₂)
	β_valid = intersect(β,axes(Yℓ₁ℓ₂n₁n₂,2))
	γ_valid = intersect(γ,axes(Yℓ₁ℓ₂n₁n₂,3))

	for γ in γ_valid
		
		Yℓ₂n₂γ = @view Yℓ₂n₂[:,γ]

		for β in β_valid
			
			Yℓ₁n₁β = @view Yℓ₁n₁[:,β]

			Yℓ₁ℓ₂n₁n₂βγ = @view Yℓ₁ℓ₂n₁n₂[:,β,γ]
			Yℓ₁ℓ₂n₁n₂₋β₋γ = @view Yℓ₁ℓ₂n₁n₂[:,-β,-γ]

			for m in m_valid

				lrange_m = l_range(lm_modes_ℓ₁ℓ₂,m)
				first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

				conjcond = -m in m_valid && -γ in γ_valid && -β in β_valid && 
					-m <= m &&  -γ <= γ && -β <= β && (β != 0 && γ != 0)

				if conjcond

					# In this case we may use the conjugation relations
					# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

					allmodes_covered = true

					for (ind,l) in enumerate(lrange_m)
						if (l,-m) ∉ lm_modes
							allmodes_covered = false
							continue
						end
						l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
						l₋mind = modeindex(lm_modes,(l,-m))
						Yℓ₁ℓ₂n₁n₂βγ[l_ind] = (-1)^(ℓ₁+ℓ₂+l+m+β+γ) * conj(Yℓ₁ℓ₂n₁n₂₋β₋γ[l₋mind])
					end

					allmodes_covered && continue
				end
				
				for m₁ in -ℓ₁:ℓ₁
		
					m₂ = m - m₁
					abs(m₂) > ℓ₂ && continue

					CG_l₁m₁_l₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

					Yℓ₁n₁βYℓ₂n₂γ = Yℓ₁n₁β[m₁]*Yℓ₂n₂γ[m₂]

					for (ind,l) in enumerate(lrange_m)
						conjcond && (l,-m) in lm_modes && continue
						l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
						Yℓ₁ℓ₂n₁n₂βγ[l_ind] += CG[l]*Yℓ₁n₁βYℓ₂n₂γ
					end
				end
			end
		end
	end

	# Specifically for m=0 the (0,0) components are purely real or imaginary
	if 0 in m_valid && 0 in β_valid && 0 in γ_valid
		lrange_m = l_range(lm_modes_ℓ₁ℓ₂,0)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),0))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			if isodd(ℓ₁+ℓ₂+l)
				# in this case the term is purely imaginary
				Yℓ₁ℓ₂n₁n₂[l_ind,0,0] = Complex(0,imag(Yℓ₁ℓ₂n₁n₂[l_ind,0,0]))
			else
				# in this case the term is purely real
				Yℓ₁ℓ₂n₁n₂[l_ind,0,0] = Complex(real(Yℓ₁ℓ₂n₁n₂[l_ind,0,0]),0)
			end
		end
	end

	return Yℓ₁ℓ₂n₁n₂
end

function BiPoSH_compute!(::OSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ₁ℓ₂n₁n₂::AbstractVector{<:Number},
	lm_modes::LM,ℓ₁::Integer,ℓ₂::Integer,
	Yℓ₁n₁::AbstractVector{<:Number},
	Yℓ₂n₂::AbstractVector{<:Number},wig3j_fn_ptr;
	w3j = zeros(ℓ₁+ℓ₂+1),CG = zeros(0:ℓ₁+ℓ₂),kwargs...)

	fill!(Yℓ₁ℓ₂n₁n₂,zero(eltype(Yℓ₁ℓ₂n₁n₂)))

	lm_modes_ℓ₁ℓ₂ = SHModes_slice(lm_modes,ℓ₁,ℓ₂)
	l_valid = l_range(lm_modes_ℓ₁ℓ₂)
	m_valid = m_range(lm_modes_ℓ₁ℓ₂)
	l_valid = l_range(lm_modes)
	m_valid = m_range(lm_modes)

	for m in m_valid
		
		lrange_m = l_range(lm_modes_ℓ₁ℓ₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		conjcond = -m in m_valid && -m < m
		
		if conjcond

			# In this case we may use the conjugation relation
			# Yʲ¹ʲ²ₗ₋ₘ = (-1)^(j₁+j₂+l+m) conj(Yʲ¹ʲ²ₗₘ)

			allmodes_covered = true

			for (ind,l) in enumerate(lrange_m)
				if (l,-m) ∉ lm_modes
					allmodes_covered = false
					continue
				end
				l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
				l₋mind = modeindex(lm_modes,(l,-m))
				Yℓ₁ℓ₂n₁n₂[l_ind] = (-1)^(ℓ₁+ℓ₂+l+m) * conj(Yℓ₁ℓ₂n₁n₂[l₋mind])
			end

			allmodes_covered && continue
		end

		for m₁ in -ℓ₁:ℓ₁
			m₂ = m - m₁
			abs(m₂) > ℓ₂ && continue

			CG_l₁m₁_l₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			Yℓ₁n₁m₁Yℓ₂n₂m₂ = Yℓ₁n₁[m₁]*Yℓ₂n₂[m₂]

			for (ind,l) in enumerate(lrange_m)
				conjcond && (l,-m) in lm_modes && continue
				l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
				Yℓ₁ℓ₂n₁n₂[l_ind] += CG[l]*Yℓ₁n₁m₁Yℓ₂n₂m₂
			end
		end

		# Specifically for m=0 the values are purely real or imaginary
		if m == 0
			for (ind,l) in enumerate(lrange_m)
				l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
				if isodd(ℓ₁+ℓ₂+l)
					# in this case the term is purely imaginary
					Yℓ₁ℓ₂n₁n₂[l_ind] = Complex(0,imag(Yℓ₁ℓ₂n₁n₂[l_ind]))
				else
					# in this case the term is purely real
					Yℓ₁ℓ₂n₁n₂[l_ind] = Complex(real(Yℓ₁ℓ₂n₁n₂[l_ind]),0)
				end
			end
		end
	end

	return Yℓ₁ℓ₂n₁n₂
end

##################################################################################################

function Wigner3j(j2,j3,m2,m3;wig3j_fn_ptr=nothing)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(-(m2 + m3))

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	exitstatus = zero(Int32)

	w3j = zeros(Float64,len)

	lib = nothing

	if @compat isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	ccall(wig3j_fn_ptr,Cvoid,
		(Ref{Float64}, 	#w3j
			Ref{Int32},	#len
			# Ref{Int32},	#jmin
			# Ref{Int32},	#jmax
			Ref{Int32},	#j2
			Ref{Int32},	#j3
			Ref{Int32},	#m1
			Ref{Int32},	#m2
			Ref{Int32},	#m3
			Ref{Int32}),#exitstatus
		w3j,len, j2, j3, m1, m2,m3, exitstatus)

	if @compat !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return w3j
end

function Wigner3j!(w3j,j2,j3,m2,m3;wig3j_fn_ptr=nothing)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(-(m2 + m3))

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	@assert(length(w3j)>=len,"length of output array must be atleast j2+j3+1=$(j2+j3+1),"*
							" supplied output array has a length of $(length(w3j))")

	exitstatus = zero(Int32)

	lib = nothing

	if @compat isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	ccall(wig3j_fn_ptr,Cvoid,
		(Ref{Float64}, 	#w3j
			Ref{Int32},	#len
			# Ref{Int32},	#jmin
			# Ref{Int32},	#jmax
			Ref{Int32},	#j2
			Ref{Int32},	#j3
			Ref{Int32},	#m1
			Ref{Int32},	#m2
			Ref{Int32},	#m3
			Ref{Int32}),#exitstatus
		w3j,len, j2, j3, m1, m2,m3, exitstatus)

	if @compat !isnothing(lib)
		Libdl.dlclose(lib)
	end
end

# Computes the Clebsch-Gordan coefficient C_{l₁m₁l₂m₂}^{lm} for all valid l
function CG_l₁m₁_l₂m₂_lm(ℓ₁,m₁,ℓ₂,m=0;wig3j_fn_ptr=nothing)
	m₂ = m-m₁
	lmin = max(abs(ℓ₁-ℓ₂),abs(m))
	lmax = ℓ₁ + ℓ₂
	w3j = Wigner3j(ℓ₁,ℓ₂,m₁,m₂;wig3j_fn_ptr=wig3j_fn_ptr)
	CG = OffsetArray(w3j[1:(lmax-lmin+1)],lmin:lmax)
	CG_l₁m₁_l₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)
	return CG
end

function CG_l₁m₁_l₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG;wig3j_fn_ptr=nothing)
	m₂ = m-m₁
	w3j = Wigner3j(ℓ₁,ℓ₂,m₁,m₂;wig3j_fn_ptr=wig3j_fn_ptr)
	CG_l₁m₁_l₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)
	return CG
end

function CG_l₁m₁_l₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=nothing)
	m₂ = m-m₁
	lmin = max(abs(ℓ₁-ℓ₂),abs(m))
	lmax = ℓ₁ + ℓ₂
	Wigner3j!(w3j,ℓ₁,ℓ₂,m₁,m₂;wig3j_fn_ptr=wig3j_fn_ptr)
	for (ind,l) in enumerate(lmin:lmax)
		CG[l] = w3j[ind]*√(2l+1)*(-1)^(ℓ₁-ℓ₂+m)
	end
	return CG
end

end

