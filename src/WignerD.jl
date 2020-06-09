module WignerD

using Compat
using Reexport
using OffsetArrays
using LinearAlgebra
using Libdl
using EllipsisNotation
using PointsOnASphere
using SphericalHarmonics
using SphericalHarmonicArrays
using WignerSymbols
using SphericalHarmonicModes
import SphericalHarmonics: Pole

import SphericalHarmonicArrays: SHArrayOneAxis
import SphericalHarmonicModes: ModeRange, SHModeRange

export Ylmn
export Ylmatrix
export Ylmatrix!
export djmatrix!
export WignerDMatrix
export WignerdMatrix
export ClampedWignerdMatrix
export BiPoSH
export BiPoSH_n1n2_n2n1
export BiPoSH!
export SphericalHarmonic
export SphericalHarmonic!
export OSH
export GSH
export PB
export Hansen
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
Base.zero(::Type{Equator}) = zero(Float64)

Base.promote_rule(::Type{Equator},::Type{Float64}) = Float64
Base.promote_rule(::Type{Equator},T::Type{<:Real}) = promote_rule(Float64,T)

#########################################################################
# Dictionary to cache the eigenvectors and eigenvalues of Jy 
#########################################################################

const JyEigenDict = Dict{Int,
	Tuple{OffsetVector{Float64,Vector{Float64}},OffsetArray{ComplexF64,2,Matrix{ComplexF64}}}}()

##########################################################################
# Struct to store d matrix and generalized Ylm accounting for symmetries
##########################################################################

struct djindices end
struct GSHindices end
struct OSHindices end

vectorinds(j::Int) = iszero(j) ? Base.IdentityUnitRange(0:0) : Base.IdentityUnitRange(-1:1)

abstract type WignerMatrices{T<:Number} <: AbstractMatrix{T} end
abstract type WignerReducedMatrix{T<:Real} <: WignerMatrices{T} end

promote_type_phase(T::Type) = promote_type(T,Int)

struct WignerdMatrix{T,V} <: WignerReducedMatrix{T}
	j :: Int
	data :: V

	function WignerdMatrix{T,V}(j::Integer,data::V) where {T<:Real,V<:AbstractArray{<:Real}}
		if length(data) < nunique(WignerdMatrix,j)
			throw(ArgumentError("array provided is not large enough"*
			" to store all values. Need an array of "*
			"length $(nunique(WignerdMatrix,j)) whereas the array provided "*
			"has length = $(length(data))"))
		end
		new{T,V}(j,data)
	end
end
nunique(::Type{<:WignerdMatrix}, j::Integer) = (j+1)^2

struct WignerDMatrix{T<:Complex,A<:Real,D<:WignerdMatrix,G<:Real} <: WignerMatrices{T}
	j :: Int
	α :: A
	data :: D
	γ :: G
end

function WignerDMatrix(T::Type,j::Integer,α::A,data::D,γ::G) where {A,G,D<:WignerdMatrix}
	WignerDMatrix{T,A,D,G}(j,α,data,γ)
end

function WignerDMatrix(j::Integer, α::A, data::WignerdMatrix{Td}, γ::G) where {A,G,Td}
	TC = Complex{promote_type(A,G)}
	T = promote_type_phase(promote_type(TC,Td))
	WignerDMatrix(T, j, α, data, γ)
end
nunique(::Type{<:WignerDMatrix},j) = nunique(WignerdMatrix,j)

function flatind(d::WignerdMatrix,m,n)
	# This is m-major, so m increases faster than n
	# Store only the left triangular quadrant
	abs(m) > d.j &&
	throw(ArgumentError("m=$m does not satisfy abs(m) ⩽ j=$(d.j)"))
	abs(n) > d.j &&
	throw(ArgumentError("n=$n does not satisfy abs(n) ⩽ j=$(d.j)"))

	# d_ij for j < n
	indskip = (2 + d.j - n)*(d.j + n)
	indskip + (m - n) + 1
end

function flatind_phase(d::WignerdMatrix,m,n)
	phase = 1
	# The left is stored
	# We evaluate the other parts using the correct phases
	if n > 0 && abs(m) <= n
		# right
		m,n = -m,-n
		phase = (-1)^(m-n)
	elseif m < -abs(n)
		# top
		m,n = n,m
		phase = (-1)^(m-n)
	elseif m > abs(n)
		# bottom
		m,n = -n,-m
	end
	ind = flatind(d,m,n)
	ind,phase
end

@inline @Base.propagate_inbounds function Base.getindex(d::WignerDMatrix{T},m::Integer,n::Integer) where {T}
	val = d.data[m,n]*cis(-m*d.α)*cis(-n*d.γ)
	convert(T, val)
end
@inline Base.@propagate_inbounds function Base.setindex!(d::WignerDMatrix,val,m::Int,n::Int)
	val /= cis(-m*d.α)*cis(-n*d.γ)
	d.data[m,n] = real(val)
end

# This struct stores only the values necessary to compute GeneralizedY
# The range of n is curtailed
struct ClampedWignerdMatrix{T,V<:AbstractArray{<:Real}} <: WignerReducedMatrix{T}
	j :: Int
	data :: V

	function ClampedWignerdMatrix{T,V}(j::Integer,data::V) where {T<:Real,V<:AbstractArray{<:Real}}
		if length(data) < nunique(ClampedWignerdMatrix,j)
			throw(ArgumentError("array provided is not large enough"*
			" to store all values. Need an array of "*
			"length $(nunique(ClampedWignerdMatrix,j)) whereas the array provided "*
			"has length = $(length(data))"))
		end
		new{T,V}(j,data)
	end
end
nunique(::Type{<:ClampedWignerdMatrix},j) = 3j+1

struct GeneralizedY{T<:Complex, V<:AbstractArray{<:Complex}} <: WignerMatrices{T}
	j :: Int
	data :: V

	function GeneralizedY{T,V}(j::Integer,data::V) where {T<:Complex,V<:AbstractArray{<:Complex}}
		if length(data) < nunique(GeneralizedY,j)
			throw(ArgumentError("array provided is not large enough"*
			" to store all values. Need an array of "*
			"length $(nunique(GeneralizedY,j)) whereas the array provided "*
			"has length = $(length(data))"))
		end
		new{T,V}(j,data)
	end
end
nunique(::Type{<:GeneralizedY},j) = (j+1)*length(vectorinds(j))

for T in [:WignerdMatrix,:ClampedWignerdMatrix,:GeneralizedY]
	@eval function $T{Q}(::UndefInitializer, j::Integer) where {Q}
		data = Vector{Q}(undef, nunique($T,j))
		$T{Q,Vector{Q}}(j, data)
	end
	@eval function $T(j::Integer, data::AbstractArray{Q}) where {Q}
		V = typeof(data)
		T = promote_type_phase(Q)
		$T{Q,V}(j, data)
	end
end

function ClampedWignerdMatrix(d::WignerdMatrix{T}) where {T}
	c = ClampedWignerdMatrix{T}(undef,d.j)
	for n in axes(c,2), m in axes(c,1)
		c[m,n] = d[m,n]
	end
	return c
end

function flatind(d::ClampedWignerdMatrix,m,n)
	# This is n-major, so n increases faster than m
	# Store only the bottom
	m >=0 || throw(ArgumentError("m needs to be >= 0"))
	vindmax = maximum(vectorinds(d.j))
	abs(n) <= vindmax || throw(ArgumentError("abs(n) needs to be <= $(vindmax)"))
	m == 0 && n != 0 && throw(ArgumentError("m = 0 only has n = 0 saved"))
	m == 0 && n == 0 && return 1
	3*(m-1) + (n + vindmax) + 2
end

function flatind(d::GeneralizedY,m,n)
	m >=0 || throw(ArgumentError("m needs to be >= 0"))
	vindmax = maximum(vectorinds(d.j))
	abs(n) <= vindmax || throw(ArgumentError("abs(n) needs to be <= $(vindmax)"))
	3m + (n + vindmax) + 1
end

function flatind_phase(d::ClampedWignerdMatrix,m,n)
	phase = 1
	if m < 0
		m,n = -m,-n
		phase = (-1)^(m-n)
	elseif m == 0 && n == -1
		m,n = 1,0
	elseif m == 0 && n == 1
		m,n = 1,0
		phase = (-1)^(m-n)
	end
	ind = flatind(d,m,n)
	ind,phase
end

function flatind_phase(d::GeneralizedY,m,n)
	phase = 1
	if m < 0
		m,n = -m,-n
		phase = (-1)^(m-n)
	end
	ind = flatind(d,m,n)
	ind,phase
end

@inline Base.@propagate_inbounds function Base.getindex(d::WignerReducedMatrix,m::Int,n::Int)
	ind,phase = flatind_phase(d,m,n)
	d.data[ind] * phase
end

@inline Base.@propagate_inbounds function Base.getindex(d::GeneralizedY{T},m::Int,n::Int) where {T}
	ind,phase = flatind_phase(d,m,n)
	val = d.data[ind]
	convert(T, m >= 0 ? val : phase*conj(val) )
end

@inline Base.@propagate_inbounds function Base.getindex(d::WignerMatrices{T}, ind::Int) where {T}
	convert(T, d.data[ind])
end

@inline Base.@propagate_inbounds function Base.setindex!(d::WignerMatrices,val,m::Int,n::Int)
	ind, phase = flatind_phase(d,m,n)
	d.data[ind] = phase * val
end
@inline Base.@propagate_inbounds Base.setindex!(d::WignerMatrices,val,ind::Int) = (d.data[ind] = val)

Base.axes(d::WignerMatrices) = (-d.j:d.j,-d.j:d.j)
Base.axes(d::WignerMatrices, dim::Integer) = axes(d)[dim]

Base.axes(d::Union{ClampedWignerdMatrix,GeneralizedY}) = (-d.j:d.j,vectorinds(d.j))

Base.size(d::WignerMatrices) = map(length,axes(d))
Base.size(d::WignerMatrices, dim::Integer) = length(axes(d,dim))

function Base.collect(d::WignerMatrices{T}) where {T}
	dfull = zeros(T,axes(d))
	for n in axes(dfull,2), m in axes(dfull,1)
		dfull[m,n] = d[m,n]
	end
	dfull
end

##########################################################################
# Wigner d matrix
##########################################################################

X(j,n) = sqrt((j+n)*(j-n+1))

Kroneckerδ(m::Integer,n::Integer) = (m == n) ? 1 : 0

function coeffi(j)
	N = 2j+1
	A = zeros(ComplexF64,N,N)
	coeffi!(j,A)
end

function coeffi!(j,A)

	N = 2j+1
	@assert(length(A)>=N^2,"array isn't long enough to store Jy")
	fill!(A,zero(eltype(A)))
	Av = reshape(@view(A[1:N^2]),N,N)
	h = Hermitian(Av)

	Av[1,1] = zero(ComplexF64)

    @inbounds for i in 1:N-1
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

	for μ in axes(λ,1)
		temp  = v[μ,m] * conj(v[μ,n])

		dj_m_n += cis(-λ[μ]*θ) * temp
	end

	dj_m_n
end

function djmatrix_terms(θ::NorthPole,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	(m == n) ? one(ComplexF64) : zero(ComplexF64)
end

function djmatrix_terms(θ::SouthPole,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	(m == -n) ? ComplexF64((-1)^(j+m)) : zero(ComplexF64)
end

function djmatrix_terms(θ::Equator,λ,v,m::Integer,n::Integer,j=div(length(λ)-1,2))
	dj_m_n = zero(ComplexF64)

	if !(isodd(j+m) && n == 0) && !(isodd(j+n) && m == 0)
		for μ in axes(λ,1)
			temp  = v[μ,m] * conj(v[μ,n])

			dj_m_n += cis(-λ[μ],θ) * temp
		end
	end

	dj_m_n
end

function djmatrix_fill!(d::ClampedWignerdMatrix,j,θ,λ,v)

	dʲ₀₀ = real(djmatrix_terms(θ,λ,v,0,0,j))
	d.data[1] = dʲ₀₀

	for m = 1:j, n = vectorinds(j)
		d[m,n] = real(djmatrix_terms(θ,λ,v,m,n,j))
	end

	return d
end

function djmatrix_fill!(d::WignerdMatrix,j,θ,λ,v)
	for n = -j:0, m = n:-n
		d[m,n] = real(djmatrix_terms(θ,λ,v,m,n,j))
	end

	return d
end

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

function djmatrix!(dj,j,θ::Real,A::Matrix{ComplexF64}=zeros(ComplexF64,2j+1,2j+1))
	λ,v = Jy_eigen!(j,A)
	djmatrix_fill!(dj,j,θ,λ,v)
end

djmatrix!(dj,j,x::SphericalPoint,args...) = djmatrix!(dj,j,x.θ,args...)

function ClampedWignerdMatrix(T::Type, j::Integer, β::Real)
	dj = ClampedWignerdMatrix{T}(undef, j)
	A = zeros(ComplexF64,2j+1,2j+1)
	djmatrix!(dj,j,β,A)
end
ClampedWignerdMatrix(j::Integer, β::Real) = ClampedWignerdMatrix(Float64, j, β)

function WignerdMatrix(T::Type, j::Integer, β::Real)
	dj = WignerdMatrix{T}(undef, j)
	A = zeros(ComplexF64,2j+1,2j+1)
	djmatrix!(dj,j,β,A)
end
WignerdMatrix(j::Integer, β::Real) = WignerdMatrix(Float64, j, β)

function WignerDMatrix(::Type{Complex{R}}, j::Integer, α::Real, β::Real, γ::Real) where {R<:Real}
	d = WignerdMatrix(R, j, β)
	WignerDMatrix(Complex{R}, j, α, d, γ)
end
WignerDMatrix(j::Integer, α::Real, β::Real, γ::Real) = WignerDMatrix(ComplexF64, j, α, β, γ)

for T in [:ClampedWignerdMatrix,:GeneralizedY]
	@eval $T(j::Integer, x::SphericalPoint) = $T(j,x.θ)
end

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

function Ylmatrix(::GSH,j::Integer,(θ,ϕ)::Tuple{Real,Real};kwargs...)
	dj_θ = ClampedWignerdMatrix(j,θ)
	Y = GeneralizedY{ComplexF64}(undef,j)
	Ylmatrix!(GSH(),Y,dj_θ,j,(θ,ϕ);compute_d_matrix=false)
end

function Ylmatrix(::OSH,l::Integer,(θ,ϕ)::Tuple{Real,Real};kwargs...)
	YSH,P,coeff = allocate_YP(OSH(),l)
	Ylmatrix!(OSH(),YSH,l,(θ,ϕ),P,coeff;kwargs...)
end

function Ylmatrix!(::GSH,Y::GeneralizedY,dj_θ::ClampedWignerdMatrix,
	j::Integer,(θ,ϕ)::Tuple{Real,Real},args...;kwargs...)

	if get(kwargs,:compute_d_matrix,true)
		djmatrix!(dj_θ,j,θ,args...)
	end

	norm = √((2j+1)/4π)

	for m in 0:j
		pre = norm * cis(m*ϕ)
		for n in vectorinds(j)
			Y[m,n] = pre * dj_θ[m,n]
		end
	end
	return Y
end

function Ylmatrix!(::GSH,Y::GeneralizedY,dj_θ::ClampedWignerdMatrix,
	j::Integer,(θ,ϕ)::Tuple{Pole,Real},args...;kwargs...)

	if get(kwargs,:compute_d_matrix,true)
		djmatrix!(dj_θ,j,θ,args...)
	end

	norm = √((2j+1)/4π)

	!iszero(ϕ) && @warn "Ignoring ϕ=$ϕ at the pole and using ϕ=0"

	for m in 0:maximum(vectorinds(j)), n in vectorinds(j)
		Y[m,n] = norm * dj_θ[m,n]
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

Ylmatrix(T::AbstractSH,l::Integer,x::SphericalPoint;kwargs...) = Ylmatrix(T,l,(x.θ,x.ϕ);kwargs...)

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
abstract type GSHType end
struct PB <: GSHType end
struct Hansen <: GSHType end

# BiPoSH Yℓ₁ℓ₂LM(n₁,n₂)
# methods for ordinary and generalized spherical harmonics
function allocate_Y₁Y₂(::OSH,j₂j₁_lmax::L₂L₁Δ;kwargs...)
	lmax = maximum(l₁_range(j₂j₁_lmax))
	l′max = maximum(l₂_range(j₂j₁_lmax))
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
function allocate_Y₁Y₂(::GSH,j₁,j₂)
	Yj₁n₁ = GeneralizedY{ComplexF64}(undef,j₁)
	Yj₂n₂ = GeneralizedY{ComplexF64}(undef,j₂)
	dj₁n₁ = ClampedWignerdMatrix{Float64}(undef,j₁)
	dj₂n₂ = ClampedWignerdMatrix{Float64}(undef,j₂)
	A = zeros(ComplexF64,2max(j₁,j₂)+1,2max(j₁,j₂)+1)
	return Yj₁n₁,Yj₂n₂,dj₁n₁,dj₂n₂,A
end
function allocate_Y₁Y₂(::GSH,lmax::Integer)
	allocate_Y₁Y₂(GSH(),lmax,lmax)
end
function allocate_Y₁Y₂(::GSH,j₂j₁_lmax::L₂L₁Δ)
	lmax = maximum(l₁_range(j₂j₁_lmax))
	l′max = maximum(l₂_range(j₂j₁_lmax))
	ll′max = max(lmax,l′max)
	allocate_Y₁Y₂(GSH(),ll′max)
end

function SHModes_slice(SHModes::SHM,ℓ′,ℓ) where {SHM<:SHModeRange}
	l_SHModes = l_range(SHModes)
	m_SHModes = m_range(SHModes)
	l_range_ℓ′ℓ = intersect(abs(ℓ-ℓ′):ℓ+ℓ′,l_SHModes)
	m_range_ℓ′ℓ = -maximum(l_range_ℓ′ℓ):maximum(l_range_ℓ′ℓ)
	m_range_ℓ′ℓ = intersect(m_range_ℓ′ℓ,m_SHModes)
	SHM(l_range_ℓ′ℓ,m_range_ℓ′ℓ)
end

function allocate_BSH(::OSH,::Any,j₂j₁_lmax::L₂L₁Δ,SHModes::SHM) where {SHM<:SHModeRange}
	T = SHVector{ComplexF64,Vector{ComplexF64},Tuple{SHM}}
	Yℓ′ℓ = SHVector{T}(undef,j₂j₁_lmax)
	for (ind,(ℓ′,ℓ)) in enumerate(j₂j₁_lmax)
		modes_section = SHModes_slice(SHModes,ℓ′,ℓ)
		Yℓ′ℓ[ind] = T(zeros(ComplexF64,length(modes_section)),
						(modes_section,),(1,))
	end
	return Yℓ′ℓ
end

function allocate_BSH(::GSH,::PB,j₂j₁_lmax::L₂L₁Δ,SHModes::SHM) where {SHM<:SHModeRange}

	ℓ′,ℓ = first(j₂j₁_lmax)

	# Need to evaluate all components to be able to swap them in the recursion
	β = Base.IdentityUnitRange(-1:1)
	γ = Base.IdentityUnitRange(-1:1)

	T = SHArray{ComplexF64,3,OffsetArray{ComplexF64,3,Array{ComplexF64,3}},
			Tuple{Base.IdentityUnitRange,Base.IdentityUnitRange,SHM},1}

	Yℓ′ℓ = SHVector{T}(undef,j₂j₁_lmax)	

	@inbounds for (ind,(ℓ′,ℓ)) in enumerate(j₂j₁_lmax)

		modes_section = SHModes_slice(SHModes,ℓ′,ℓ)
		Yℓ′ℓ[ind] = T(zeros(ComplexF64,β,γ,length(modes_section)),
			(β,γ,modes_section),(3,))
	end
	return Yℓ′ℓ
end

function allocate_BSH(::GSH,::Hansen,j₂j₁_lmax::L₂L₁Δ,SHModes::SHM) where {SHM<:SHModeRange}

	ℓ′,ℓ = first(j₂j₁_lmax)

	# Need to evaluate all components to be able to swap them in the recursion
	β = Base.IdentityUnitRange(0:1)
	γ = Base.IdentityUnitRange(0:1)

	T = SHArray{ComplexF64,3,OffsetArray{ComplexF64,3,Array{ComplexF64,3}},
			Tuple{Base.IdentityUnitRange,Base.IdentityUnitRange,SHM},1}

	Yℓ′ℓ = SHVector{T}(undef,j₂j₁_lmax)	

	@inbounds for (ind,(ℓ′,ℓ)) in enumerate(j₂j₁_lmax)

		modes_section = SHModes_slice(SHModes,ℓ′,ℓ)
		Yℓ′ℓ[ind] = T(zeros(ComplexF64,β,γ,length(modes_section)),
			(β,γ,modes_section),(3,))
	end
	return Yℓ′ℓ
end

function allocate_BSH(ASH::AbstractSH,GSHT,ℓ_range::AbstractUnitRange,
	SHModes::SHModeRange)

	j₂j₁_lmax = L₂L₁Δ(ℓ_range,SHModes)
	allocate_BSH(ASH,GSHT,j₂j₁_lmax,SHModes)
end

function BiPoSH(ASH::AbstractSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	l::Integer,m::Integer,j₁::Integer,j₂::Integer;kwargs...)
	
	@assert(δ(j₁,j₂,l),"|j₁-j₂|<=l<=j₁+j₂ not satisfied for j₁=$j₁, j₂=$j₂ and l=$l")
	@assert(abs(m)<=l,"abs(m)<=l not satisfied for m=$m and l=$l")

	Yℓ₁ℓ₂n₁n₂ = BiPoSH(ASH,GSHT,x1,x2,LM(l:l,m:m),j₁,j₂;kwargs...)
	Yℓ₁ℓ₂n₁n₂[..,1]
end

function BiPoSH(::GSH,::PB,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,j₁::Integer,j₂::Integer,args...;kwargs...)

	modes_section = SHModes_slice(SHModes,j₁,j₂)
	B = SHArray(Base.IdentityUnitRange(-1:1),Base.IdentityUnitRange(-1:1),modes_section)
	Y_and_d_arrs = allocate_Y₁Y₂(GSH(),j₁,j₂)
	j₁₂max = max(j₁,j₂)

	BiPoSH!(GSH(),PB(),x1,x2,B,j₁,j₂,Y_and_d_arrs...,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

function BiPoSH(::GSH,::Hansen,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,j₁::Integer,j₂::Integer,args...;kwargs...)

	modes_section = SHModes_slice(SHModes,j₁,j₂)
	B = SHArray(Base.IdentityUnitRange(0:1),Base.IdentityUnitRange(0:1),modes_section)
	Y_and_d_arrs = allocate_Y₁Y₂(GSH(),j₁,j₂)
	j₁₂max = max(j₁,j₂)

	BiPoSH!(GSH(),Hansen(),x1,x2,B,j₁,j₂,Y_and_d_arrs...,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

function BiPoSH(::OSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,j₁::Integer,j₂::Integer,
	args...;kwargs...)

	modes_section = SHModes_slice(SHModes,j₁,j₂)
	B = SHVector(modes_section)
	Y_and_d_arrs = allocate_Y₁Y₂(OSH(),max(j₁,j₂))

	BiPoSH!(OSH(),GSHT,x1,x2,B,j₁,j₂,Y_and_d_arrs...,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

function BiPoSH(ASH::AbstractSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHM,j₂j₁_lmax::L₂L₁Δ,
	args...;kwargs...) where {SHM<:SHModeRange}

	Yℓ′n₁ℓn₂ = allocate_BSH(ASH,GSHT,j₂j₁_lmax,SHModes)
	Y_and_d_arrs = allocate_Y₁Y₂(ASH,j₂j₁_lmax)

	BiPoSH!(ASH,GSHT,x1,x2,Yℓ′n₁ℓn₂,j₂j₁_lmax,Y_and_d_arrs...,args...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

BiPoSH(ASH::AbstractSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHModeRange,ℓ_range::AbstractUnitRange,args...;kwargs...) = 
	BiPoSH(ASH,GSHT,x1,x2,SHModes,L₂L₁Δ(ℓ_range,SHModes),args...;kwargs...)

BiPoSH(ASH::AbstractSH,GSHT,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH(ASH,GSHT,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

function BiPoSH_n1n2_n2n1(ASH::AbstractSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHM,j₂j₁_lmax::L₂L₁Δ;kwargs...) where {SHM<:SHModeRange}

	Yℓ′n₁ℓn₂ = allocate_BSH(ASH,GSHT,j₂j₁_lmax,SHModes)
	Yℓ′n₂ℓn₁ = allocate_BSH(ASH,GSHT,j₂j₁_lmax,SHModes)

	Y_and_d_arrs = allocate_Y₁Y₂(ASH,j₂j₁_lmax)

	BiPoSH!(ASH,GSHT,x1,x2,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,Y_and_d_arrs...;
		kwargs...,compute_Y₁=true,compute_Y₂=true)
end

BiPoSH_n1n2_n2n1(ASH::AbstractSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	SHModes::SHM,ℓ_range::AbstractUnitRange,args...;kwargs...) where {SHM<:SHModeRange} = 
	BiPoSH_n1n2_n2n1(ASH,GSHT,x1,x2,SHModes,L₂L₁Δ(ℓ_range,SHModes),args...;kwargs...)

BiPoSH_n1n2_n2n1(ASH::AbstractSH,GSHT,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH_n1n2_n2n1(ASH,GSHT,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

# For one (ℓ,ℓ′) just unpack the x1,x2 into (θ,ϕ) and and call BiPoSH_compute!
# These methods compute the monopolar YSH and pass them on to BiPoSH_compute!
function BiPoSH!(::OSH,::Any,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
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
			Yℓ₁n₁,Yℓ₂n₂,wig3j_fn_ptr,w3j,CG)
	finally
		Libdl.dlclose(lib)
	end

	return B
end

function BiPoSH!(::GSH,GSHT::GSHType,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	B::AbstractArray{<:Complex,3},
	SHModes::SHModeRange,
	ℓ₁::Integer,ℓ₂::Integer,
	Yℓ₁n₁::GeneralizedY,
	Yℓ₂n₂::GeneralizedY,
	dℓ₁n₁::ClampedWignerdMatrix,
	dℓ₂n₂::ClampedWignerdMatrix,
	A_djcoeffi = zeros(ComplexF64,2max(ℓ₁,ℓ₂)+1,2max(ℓ₁,ℓ₂)+1);
	compute_Y₁=true,compute_Y₂=true,
	w3j = zeros(ℓ₁+ℓ₂+1),CG = zeros(0:ℓ₁+ℓ₂),kwargs...)

	compute_Y₁ && Ylmatrix!(GSH(),Yℓ₁n₁,dℓ₁n₁,ℓ₁,(θ₁,ϕ₁),A_djcoeffi)
	compute_Y₂ && Ylmatrix!(GSH(),Yℓ₂n₂,dℓ₂n₂,ℓ₂,(θ₂,ϕ₂),A_djcoeffi,
					# don't recompute dj if unnecessary
					compute_d_matrix = !( compute_Y₁ && (ℓ₁ == ℓ₂) && (θ₁ == θ₂) && (dℓ₁n₁ === dℓ₂n₂) ) )

	lib = nothing
	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do 
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		BiPoSH_compute!(GSH(),GSHT,(θ₁,ϕ₁),(θ₂,ϕ₂),B,SHModes,ℓ₁,ℓ₂,
			Yℓ₁n₁,Yℓ₂n₂,wig3j_fn_ptr,w3j,CG)
	finally
		Libdl.dlclose(lib)
	end

	return B
end

BiPoSH!(ASH::AbstractSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	B::SHArrayOneAxis,ℓ₁::Integer,args...;kwargs...) = 
	BiPoSH!(ASH,GSHT,x1,x2,B,shmodes(B),ℓ₁,args...;kwargs...)

BiPoSH!(ASH::AbstractSH,GSHT,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH!(ASH,GSHT,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

"""
	BiPoSH!(OSH(),(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂::AbstractVector{<:SHVector},
	SHModes::SHModeRange,j₂j₁_lmax::L₂L₁Δ,args...;kwargs...)
	Compute BiPoSH for a range in ℓ and ℓ′
"""
function BiPoSH!(::OSH,GSHT,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHVector},
	j₂j₁_lmax::L₂L₁Δ,
	YSH_n₁::AbstractVector{<:Complex},
	YSH_n₂::AbstractVector{<:Complex},
	P::AbstractVector{<:Real},coeff;
	CG = zeros( 0:(maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax))) ),
	w3j = zeros( maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax)) + 1),
	compute_Y₁=true,compute_Y₂=true,kwargs...)

	lmax = maximum(l₁_range(j₂j₁_lmax))
	l′max = maximum(l₂_range(j₂j₁_lmax))

	compute_Y₁ && compute_YP!(l′max,(θ₁,ϕ₁),YSH_n₁,P,coeff)
	compute_Y₂ && compute_YP!(lmax,(θ₂,ϕ₂),YSH_n₂,P,coeff)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(j₂j₁_lmax)

			BiPoSH!(OSH(),GSHT,(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂[ℓ′ℓind],
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

function BiPoSH!(::GSH,GSHT::GSHType,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHArray{<:Number,3}},
	j₂j₁_lmax::L₂L₁Δ,
	Yℓ₁n₁::GeneralizedY,
	Yℓ₂n₂::GeneralizedY,
	dℓ₁n₁::ClampedWignerdMatrix,
	dℓ₂n₂::ClampedWignerdMatrix,
	A_djcoeffi = begin
		lmax = maximum(l₁_range(j₂j₁_lmax))
		l′max = maximum(l₂_range(j₂j₁_lmax))
		l′lmax = max(lmax,l′max)
		zeros(ComplexF64,2l′lmax+1,2l′lmax+1)
	end;
	CG = zeros( 0:(maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax))) ),
	w3j = zeros( maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax)) + 1),
	compute_Y₁=true,compute_Y₂=true,kwargs...)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (ℓ′ℓind,(ℓ′,ℓ)) in enumerate(j₂j₁_lmax)

			BiPoSH!(GSH(),GSHT,(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂[ℓ′ℓind],
				shmodes(Yℓ′n₁ℓn₂[ℓ′ℓind]),ℓ′,ℓ,
				Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,dℓ₂n₂,A_djcoeffi;
				CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
				compute_Y₁=compute_Y₁,compute_Y₂=compute_Y₂)
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂
end

BiPoSH!(::OSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHVector},
	Y1::AbstractVector{<:Complex},args...;kwargs...) = 
	BiPoSH!(OSH(),GSHT,x1,x2,Yℓ′n₁ℓn₂,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

BiPoSH!(::GSH,GSHT::GSHType,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHArrayOneAxis},
	Y1::AbstractMatrix{<:Complex},args...;kwargs...) = 
	BiPoSH!(GSH(),GSHT,x1,x2,Yℓ′n₁ℓn₂,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

"""
	BiPoSH!(OSH(),Yℓ′n₁ℓn₂::AbstractVector{<:SHVector},
	Yℓ′n₂ℓn₁::AbstractVector{<:SHVector},
	SHModes::SHModeRange,j₂j₁_lmax::L₂L₁Δ,args...;kwargs...)
	Compute BiPoSH for a range in ℓ and ℓ′ by switching the two points
	Returns Yℓ′n₁ℓn₂ and Yℓ′n₂ℓn₁
"""
function BiPoSH!(::OSH,GSHT,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHVector{<:Number}},
	Yℓ′n₂ℓn₁::AbstractVector{<:SHVector{<:Number}},
	j₂j₁_lmax::L₂L₁Δ,
	YSH_n₁::AbstractVector{<:Number},
	YSH_n₂::AbstractVector{<:Number},
	P::AbstractVector{<:Real},coeff;
	CG = zeros( 0:(maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax))) ),
	w3j = zeros( maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax)) + 1),
	compute_Y₁=true,
	compute_Y₂=true,kwargs...)

	lmax = maximum(l₁_range(j₂j₁_lmax))
	l′max = maximum(l₂_range(j₂j₁_lmax))
	ll′max = max(lmax,l′max)

	compute_Y₁ && compute_YP!(ll′max,(θ₁,ϕ₁),YSH_n₁,P,coeff)
	compute_Y₂ && compute_YP!(ll′max,(θ₂,ϕ₂),YSH_n₂,P,coeff)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (indℓ′ℓ,(ℓ′,ℓ)) in enumerate(j₂j₁_lmax)

			# In one pass we can compute Yℓ′n₁ℓn₂ and Yℓn₂ℓ′n₁

			@inbounds Yℓ′n₁ℓn₂_st = Yℓ′n₁ℓn₂[indℓ′ℓ]
			@inbounds Yℓ′n₂ℓn₁_st = Yℓ′n₂ℓn₁[indℓ′ℓ]

			# We use Yℓ′n₂ℓn₁ = (-1)^(ℓ+ℓ′+s) Yℓn₁ℓ′n₂
			# and Yℓ′n₁ℓn₂ = (-1)^(ℓ+ℓ′+s) Yℓn₂ℓ′n₁
			# Precomputation of the RHS would have happened if ℓ′<ℓ, 
			# as the modes are sorted in order of increasing ℓ

			if (ℓ,ℓ′) in j₂j₁_lmax && ℓ′<ℓ
				# In this case Yℓn₁ℓ′n₂ and Yℓn₂ℓ′n₁ have already been computed
				# This means we can evaluate Yℓ′n₂ℓn₁ and Yℓ′n₁ℓn₂ using the formulae
				# presented above

				indℓℓ′ = modeindex(j₂j₁_lmax,(ℓ,ℓ′))
				@inbounds Yℓn₁ℓ′n₂_st = Yℓ′n₁ℓn₂[indℓℓ′]
				@inbounds Yℓn₂ℓ′n₁_st = Yℓ′n₂ℓn₁[indℓℓ′]

				@inbounds for (indst,(s,t)) in enumerate(shmodes(Yℓ′n₂ℓn₁_st))
					Yℓ′n₂ℓn₁_st[indst] = (-1)^(ℓ+ℓ′+s)*Yℓn₁ℓ′n₂_st[indst]
					Yℓ′n₁ℓn₂_st[indst] = (-1)^(ℓ+ℓ′+s)*Yℓn₂ℓ′n₁_st[indst]
				end
			else
				# Default case, where we need to evaluate both

				BiPoSH!(OSH(),GSHT,(θ₂,ϕ₂),(θ₁,ϕ₁),Yℓ′n₂ℓn₁_st,ℓ′,ℓ,
					YSH_n₂,YSH_n₁,P,coeff;
					CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=!compute_Y₁,compute_Y₂=!compute_Y₂)

				BiPoSH!(OSH(),GSHT,(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ′n₁ℓn₂_st,ℓ′,ℓ,
					YSH_n₁,YSH_n₂,P,coeff;
					CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=!compute_Y₁,compute_Y₂=!compute_Y₂)
			end
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁
end

function BiPoSH!(::GSH,GSHT::GSHType,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::AbstractVector{<:SHArray{<:Number,3}},
	Yℓ′n₂ℓn₁::AbstractVector{<:SHArray{<:Number,3}},
	j₂j₁_lmax::L₂L₁Δ,
	Yℓ₁n₁::GeneralizedY,
	Yℓ₂n₂::GeneralizedY,
	dℓ₁n₁::ClampedWignerdMatrix,
	dℓ₂n₂::ClampedWignerdMatrix,
	A_djcoeffi = begin
		lmax = maximum(l₁_range(j₂j₁_lmax))
		l′max = maximum(l₂_range(j₂j₁_lmax))
		l′lmax = max(lmax,l′max)
		zeros(ComplexF64,2l′lmax+1,2l′lmax+1)
	end;
	CG = zeros( 0:(maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax))) ),
	w3j = zeros( maximum(l₁_range(j₂j₁_lmax)) + maximum(l₂_range(j₂j₁_lmax)) + 1),
	compute_Y₁=true,compute_Y₂=true,kwargs...)

	lib = nothing

	try
		wig3j_fn_ptr = get(kwargs,:wig3j_fn_ptr) do
			lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
			Libdl.dlsym(lib,:wigner3j_wrapper)
		end

		for (ind_j₂j₁,(j₂,j₁)) in enumerate(j₂j₁_lmax)

			# In one pass we can compute Yℓ′n₁ℓn₂ and Yℓn₂ℓ′n₁

			@inbounds Pʲ²ʲ¹ₗₘ_n₁n₂ = Yℓ′n₁ℓn₂[ind_j₂j₁]
			@inbounds Pʲ²ʲ¹ₗₘ_n₂n₁ = Yℓ′n₂ℓn₁[ind_j₂j₁]

			lm_iter = shmodes(Pʲ²ʲ¹ₗₘ_n₁n₂)

			# We use the relation between the helicity basis components 
			# Pʲ²ʲ¹ₗₘ_α₂α₁(n₁,n₂) = (-1)ʲ¹⁺ʲ²⁺ˡ Pʲ¹ʲ²ₗₘ_α₁α₂(n₂,n₁)
			# and Pʲ²ʲ¹ₗₘ_α₂α₁(n₂,n₁) = (-1)ʲ¹⁺ʲ²⁺ˡ Pʲ¹ʲ²ₗₘ_α₁α₂(n₁,n₂)
			# Precomputation of the RHS would have happened if j₂ < j₁, 
			# as the modes are sorted in order of increasing j₁

			if (j₁,j₂) in j₂j₁_lmax && j₂ < j₁ && !iszero(j₂)
				# In this case Pʲ¹ʲ²ₗₘ_α₁α₂(n₁,n₂) and Pʲ¹ʲ²ₗₘ_α₁α₂(n₂,n₁) have already been computed
				# This means we may evaluate Pʲ²ʲ¹ₗₘ_α₂α₁(n₂,n₁) and Pʲ²ʲ¹ₗₘ_α₂α₁(n₁,n₂) using the formulae
				# presented above

				ind_j₁j₂ = modeindex(j₂j₁_lmax,(j₁,j₂))

				@inbounds Pʲ¹ʲ²ₗₘ_n₁n₂ = Yℓ′n₁ℓn₂[ind_j₁j₂]
				@inbounds Pʲ¹ʲ²ₗₘ_n₂n₁ = Yℓ′n₂ℓn₁[ind_j₁j₂]

				@inbounds for (ind_lm,(l,m)) in enumerate(lm_iter)
					phase = (-1)^(j₁+j₂+l)
					for α₂ in axes(Pʲ¹ʲ²ₗₘ_n₁n₂,2), α₁ in axes(Pʲ¹ʲ²ₗₘ_n₁n₂,1)
						Pʲ²ʲ¹ₗₘ_n₂n₁[α₂,α₁,ind_lm] = phase * Pʲ¹ʲ²ₗₘ_n₁n₂[α₁,α₂,ind_lm]
						Pʲ²ʲ¹ₗₘ_n₁n₂[α₂,α₁,ind_lm] = phase * Pʲ¹ʲ²ₗₘ_n₂n₁[α₁,α₂,ind_lm]
					end
				end
			else
				# Default case, where we need to evaluate both

				d2 = (j₁ == j₂) && (θ₁ == θ₂) ? dℓ₁n₁ : dℓ₂n₂

				BiPoSH!(GSH(),GSHT,(θ₂,ϕ₂),(θ₁,ϕ₁),Pʲ²ʲ¹ₗₘ_n₂n₁,j₂,j₁,
					Yℓ₂n₂,Yℓ₁n₁,d2,dℓ₁n₁,A_djcoeffi;
					kwargs...,CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=true,compute_Y₂=true)

				BiPoSH!(GSH(),GSHT,(θ₁,ϕ₁),(θ₂,ϕ₂),Pʲ²ʲ¹ₗₘ_n₁n₂,j₂,j₁,
					Yℓ₁n₁,Yℓ₂n₂,dℓ₁n₁,d2,A_djcoeffi;
					kwargs...,CG=CG,w3j=w3j,wig3j_fn_ptr=wig3j_fn_ptr,
					compute_Y₁=(j₁ != j₂),compute_Y₂=(j₁ != j₂))
			end
		end
	finally
		Libdl.dlclose(lib)
	end

	return Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁
end

@inline BiPoSH!(::OSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHVector},Yℓ′n₂ℓn₁::SHVector{<:SHVector},
	Y1::AbstractVector{<:Complex},args...;kwargs...) = 
	BiPoSH!(OSH(),GSHT,x1,x2,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

@inline BiPoSH!(::GSH,GSHT,x1::Tuple{Real,Real},x2::Tuple{Real,Real},
	Yℓ′n₁ℓn₂::SHVector{<:SHArrayOneAxis},
	Yℓ′n₂ℓn₁::SHVector{<:SHArrayOneAxis},
	Y1::AbstractMatrix{<:Complex},args...;kwargs...) = 
	BiPoSH!(GSH(),GSHT,x1,x2,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,shmodes(Yℓ′n₁ℓn₂),Y1,args...;kwargs...)

# The actual functions that do the calculation for one pair of (ℓ₁,ℓ₂) and 
# (θ₁,ϕ₁) and (θ₂,ϕ₂). The BiPoSH! functions call these.
# We assume that the monopolar YSH are already computed in the BiPoSH! calls
function BiPoSH_compute!(::GSH,::PB,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yj₁j₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,j₁::Integer,j₂::Integer,
	Yj₁n₁::GeneralizedY,Yj₂n₂::GeneralizedY,wig3j_fn_ptr,
	w3j = zeros(j₁+j₂+1),CG = zeros(0:j₁+j₂))

	fill!(Yj₁j₂n₁n₂,zero(eltype(Yj₁j₂n₁n₂)))

	lm_modes_j₁j₂ = SHModes_slice(lm_modes,j₁,j₂)
	l_valid = l_range(lm_modes_j₁j₂)
	m_valid = m_range(lm_modes_j₁j₂)
	β_valid = vectorinds(j₁)
	γ_valid = vectorinds(j₂)

	m_compute = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? m_valid : (0:maximum(m_valid))
	m_symmetry = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? (0:-1) : (minimum(m_valid):-1)

	@inbounds for m in m_compute

		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for m₁ in -j₁:j₁

			m₂ = m - m₁
			abs(m₂) > j₂ && continue

			CG_j₁m₁_j₂m₂_lm!(j₁,m₁,j₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			for (ind,l) in enumerate(lrange_m)

				l_ind = (ind - 1) + first_l_ind # l's increase faster than m

				Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ = CG[l]
				iszero(Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂) && continue

				for γ in γ_valid, β in β_valid
					Yj₁j₂n₁n₂[β,γ,l_ind] += Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ * Yj₁n₁[m₁,β] * Yj₂n₂[m₂,γ]
				end
			end
		end
	end

	# Specifically for m=0 the (0,0) components are purely real or imaginary
	if 0 in m_valid
		@inbounds for l in l_range(lm_modes_j₁j₂,0)
			l_ind = modeindex(lm_modes,(l,0))
			if isodd(j₁+j₂+l)
				# in this case the term is purely imaginary
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,0,l_ind]))
			else
				# in this case the term is purely real
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,0,l_ind]),0)
			end
		end
	end

	@inbounds for m in m_symmetry
		
		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			l₋mind = modeindex(lm_modes,(l,-m))
			
			for γ in γ_valid, β in β_valid
				
				# In this case we may use the conjugation relations
				# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

				Yj₁j₂n₁n₂[β,γ,l_ind] = (-1)^(j₁+j₂+l+m+β+γ) * conj(Yj₁j₂n₁n₂[-β,-γ,l₋mind])
			end
		end
	end

	return Yj₁j₂n₁n₂
end

function BiPoSH_compute!(::OSH,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yℓ₁ℓ₂n₁n₂::AbstractVector{<:Number},
	lm_modes::LM,ℓ₁::Integer,ℓ₂::Integer,
	Yℓ₁n₁::AbstractVector{<:Number},
	Yℓ₂n₂::AbstractVector{<:Number},wig3j_fn_ptr,
	w3j = zeros(ℓ₁+ℓ₂+1),CG = zeros(0:ℓ₁+ℓ₂))

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

			CG_j₁m₁_j₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

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

Hansen_phase(j,m,α) = (1+(-1)^(j+m))*2^abs(α)/2

function BiPoSH_compute!(::GSH,::Hansen,(θ₁,ϕ₁)::Tuple{Equator,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yj₁j₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,j₁::Integer,j₂::Integer,
	Yj₁n₁::GeneralizedY,Yj₂n₂::GeneralizedY,wig3j_fn_ptr,
	w3j = zeros(j₁+j₂+1),CG = zeros(0:j₁+j₂))

	fill!(Yj₁j₂n₁n₂,zero(eltype(Yj₁j₂n₁n₂)))

	lm_modes_j₁j₂ = SHModes_slice(lm_modes,j₁,j₂)
	l_valid = l_range(lm_modes_j₁j₂)
	m_valid = m_range(lm_modes_j₁j₂)
	β_valid = intersect(0:1,vectorinds(j₁)) # the -1 term is accounted for by the phase factor
	γ_valid = vectorinds(j₂)

	m_compute = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? m_valid : (0:maximum(m_valid))
	m_symmetry = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? (0:-1) : (minimum(m_valid):-1)

	@inbounds for m in m_compute

		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for m₁ in -j₁:2:j₁

			m₂ = m - m₁
			abs(m₂) > j₂ && continue

			CG_j₁m₁_j₂m₂_lm!(j₁,m₁,j₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			for (ind,l) in enumerate(lrange_m)

				l_ind = (ind - 1) + first_l_ind # l's increase faster than m

				Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ = CG[l]
				iszero(Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂) && continue

				for γ in γ_valid, β in β_valid
					Yj₁j₂n₁n₂[abs(β),abs(γ),l_ind] += Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ * Yj₁n₁[m₁,β] * Yj₂n₂[m₂,γ] *
														Hansen_phase(j₁,m₁,β)
				end
			end
		end
	end

	@inbounds for m in m_symmetry
		
		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			l₋mind = modeindex(lm_modes,(l,-m))
			
			for γ in axes(Yj₁j₂n₁n₂,2), β in axes(Yj₁j₂n₁n₂,1)
				
				# In this case we may use the conjugation relations
				# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

				Yj₁j₂n₁n₂[β,γ,l_ind] = (-1)^(j₁+j₂+l+m+β+γ) * conj(Yj₁j₂n₁n₂[β,γ,l₋mind])
			end
		end
	end

	# Specifically for m=0 all the components are either real or purely imaginary
	if 0 in m_valid
		@inbounds for l in l_range(lm_modes_j₁j₂,0)
			l_ind = modeindex(lm_modes,(l,0))
			if isodd(j₁+j₂+l)
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,0,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,1,l_ind]),0)
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,1,l_ind]))
			else
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,1,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,0,l_ind]))
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,1,l_ind]),0)
			end
		end
	end

	return Yj₁j₂n₁n₂
end

function BiPoSH_compute!(::GSH,::Hansen,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Equator,Real},
	Yj₁j₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,j₁::Integer,j₂::Integer,
	Yj₁n₁::GeneralizedY,Yj₂n₂::GeneralizedY,wig3j_fn_ptr,
	w3j = zeros(j₁+j₂+1),CG = zeros(0:j₁+j₂))

	fill!(Yj₁j₂n₁n₂,zero(eltype(Yj₁j₂n₁n₂)))

	lm_modes_j₁j₂ = SHModes_slice(lm_modes,j₁,j₂)
	l_valid = l_range(lm_modes_j₁j₂)
	m_valid = m_range(lm_modes_j₁j₂)
	β_valid = vectorinds(j₁)
	γ_valid = intersect(0:1,vectorinds(j₂)) # the -1 term is accounted for by the phase factor

	m_compute = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? m_valid : (0:maximum(m_valid))
	m_symmetry = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? (0:-1) : (minimum(m_valid):-1)

	@inbounds for m in m_compute

		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for m₂ in -j₂:2:j₂

			m₁ = m - m₂
			abs(m₁) > j₁ && continue

			CG_j₁m₁_j₂m₂_lm!(j₁,m₁,j₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			for (ind,l) in enumerate(lrange_m)

				l_ind = (ind - 1) + first_l_ind # l's increase faster than m

				Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ = CG[l]
				iszero(Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂) && continue

				for γ in γ_valid, β in β_valid
					Yj₁j₂n₁n₂[abs(β),abs(γ),l_ind] += Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ * Yj₁n₁[m₁,β] * Yj₂n₂[m₂,γ] * 
														Hansen_phase(j₂,m₂,γ)
				end
			end
		end
	end

	@inbounds for m in m_symmetry
		
		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			l₋mind = modeindex(lm_modes,(l,-m))
			
			for γ in axes(Yj₁j₂n₁n₂,2), β in axes(Yj₁j₂n₁n₂,1)
				
				# In this case we may use the conjugation relations
				# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

				Yj₁j₂n₁n₂[β,γ,l_ind] = (-1)^(j₁+j₂+l+m+β+γ) * conj(Yj₁j₂n₁n₂[β,γ,l₋mind])
			end
		end
	end

	# Specifically for m=0 all the components are either real or purely imaginary
	if 0 in m_valid
		@inbounds for l in l_range(lm_modes_j₁j₂,0)
			l_ind = modeindex(lm_modes,(l,0))
			if isodd(j₁+j₂+l)
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,0,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,1,l_ind]),0)
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,1,l_ind]))
			else
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,1,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,0,l_ind]))
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,1,l_ind]),0)
			end
		end
	end

	return Yj₁j₂n₁n₂
end

function BiPoSH_compute!(::GSH,::Hansen,(θ₁,ϕ₁)::Tuple{Equator,Real},(θ₂,ϕ₂)::Tuple{Equator,Real},
	Yj₁j₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,j₁::Integer,j₂::Integer,
	Yj₁n₁::GeneralizedY,Yj₂n₂::GeneralizedY,wig3j_fn_ptr,
	w3j = zeros(j₁+j₂+1),CG = zeros(0:j₁+j₂))

	fill!(Yj₁j₂n₁n₂,zero(eltype(Yj₁j₂n₁n₂)))

	lm_modes_j₁j₂ = SHModes_slice(lm_modes,j₁,j₂)
	l_valid = l_range(lm_modes_j₁j₂)
	m_valid = m_range(lm_modes_j₁j₂)
	β_valid = intersect(0:1,vectorinds(j₁)) # the -1 term is accounted for by the phase factor
	γ_valid = intersect(0:1,vectorinds(j₂)) # the -1 term is accounted for by the phase factor

	m_compute = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? m_valid : (0:maximum(m_valid))
	m_symmetry = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? (0:-1) : (minimum(m_valid):-1)

	@inbounds for m in m_compute

		isodd(j₁ + j₂) && iseven(m) && continue

		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for m₁ in -j₁:2:j₁

			m₂ = m - m₁
			abs(m₂) > j₂ && continue
			isodd(j₂ + m₂) && continue

			CG_j₁m₁_j₂m₂_lm!(j₁,m₁,j₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			for (ind,l) in enumerate(lrange_m)

				l_ind = (ind - 1) + first_l_ind # l's increase faster than m

				Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ = CG[l]
				iszero(Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂) && continue

				for γ in γ_valid, β in β_valid
					Yj₁j₂n₁n₂[β,γ,l_ind] += Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ * Yj₁n₁[m₁,β] * Yj₂n₂[m₂,γ] * 
											Hansen_phase(j₁,m₁,β) * Hansen_phase(j₂,m₂,γ)
				end
			end
		end
	end

	@inbounds for m in m_symmetry

		isodd(j₁+j₂) && iseven(m) && continue
		
		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			l₋mind = modeindex(lm_modes,(l,-m))
			
			for γ in axes(Yj₁j₂n₁n₂,2), β in axes(Yj₁j₂n₁n₂,1)
				
				# In this case we may use the conjugation relations
				# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

				Yj₁j₂n₁n₂[β,γ,l_ind] = (-1)^(j₁+j₂+l+m+β+γ) * conj(Yj₁j₂n₁n₂[β,γ,l₋mind])
			end
		end
	end

	# Specifically for m=0 all the components are either real or purely imaginary
	if 0 in m_valid
		@inbounds for l in l_range(lm_modes_j₁j₂,0)
			l_ind = modeindex(lm_modes,(l,0))
			if isodd(j₁+j₂+l)
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,0,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,1,l_ind]),0)
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,1,l_ind]))
			else
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,1,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,0,l_ind]))
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,1,l_ind]),0)
			end
		end
	end

	return Yj₁j₂n₁n₂
end

function BiPoSH_compute!(::GSH,::Hansen,(θ₁,ϕ₁)::Tuple{Real,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yj₁j₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,j₁::Integer,j₂::Integer,
	Yj₁n₁::GeneralizedY,Yj₂n₂::GeneralizedY,wig3j_fn_ptr,
	w3j = zeros(j₁+j₂+1),CG = zeros(0:j₁+j₂))

	fill!(Yj₁j₂n₁n₂,zero(eltype(Yj₁j₂n₁n₂)))

	lm_modes_j₁j₂ = SHModes_slice(lm_modes,j₁,j₂)
	l_valid = l_range(lm_modes_j₁j₂)
	m_valid = m_range(lm_modes_j₁j₂)
	β_valid = vectorinds(j₁)
	γ_valid = vectorinds(j₂)

	m_compute = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? m_valid : (0:maximum(m_valid))
	m_symmetry = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? (0:-1) : (minimum(m_valid):-1)

	@inbounds for m in m_compute

		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for m₁ in -j₁:j₁

			m₂ = m - m₁
			abs(m₂) > j₂ && continue

			CG_j₁m₁_j₂m₂_lm!(j₁,m₁,j₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			for (ind,l) in enumerate(lrange_m)

				l_ind = (ind - 1) + first_l_ind # l's increase faster than m

				Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ = CG[l]
				iszero(Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂) && continue

				for γ in γ_valid, β in β_valid
					Yj₁j₂n₁n₂[abs(β),abs(γ),l_ind] += Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ * Yj₁n₁[m₁,β] * Yj₂n₂[m₂,γ]
				end
			end
		end
	end

	@inbounds for m in m_symmetry
		
		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			l₋mind = modeindex(lm_modes,(l,-m))
			
			for γ in axes(Yj₁j₂n₁n₂,2), β in axes(Yj₁j₂n₁n₂,1)
				
				# In this case we may use the conjugation relations
				# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

				Yj₁j₂n₁n₂[β,γ,l_ind] = (-1)^(j₁+j₂+l+m+β+γ) * conj(Yj₁j₂n₁n₂[β,γ,l₋mind])
			end
		end
	end

	# Specifically for m=0 all the components are either real or purely imaginary
	if 0 in m_valid
		@inbounds for l in l_range(lm_modes_j₁j₂,0)
			l_ind = modeindex(lm_modes,(l,0))
			if isodd(j₁+j₂+l)
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,0,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,1,l_ind]),0)
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,1,l_ind]))
			else
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,1,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,0,l_ind]))
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,1,l_ind]),0)
			end
		end
	end

	return Yj₁j₂n₁n₂
end

function BiPoSH_compute!(::GSH,::Hansen,(θ₁,ϕ₁)::Tuple{NorthPole,Real},(θ₂,ϕ₂)::Tuple{Real,Real},
	Yj₁j₂n₁n₂::AbstractArray{<:Number,3},
	lm_modes::LM,j₁::Integer,j₂::Integer,
	Yj₁n₁::GeneralizedY,Yj₂n₂::GeneralizedY,wig3j_fn_ptr,
	w3j = zeros(j₁+j₂+1),CG = zeros(0:j₁+j₂))

	fill!(Yj₁j₂n₁n₂,zero(eltype(Yj₁j₂n₁n₂)))

	lm_modes_j₁j₂ = SHModes_slice(lm_modes,j₁,j₂)
	l_valid = l_range(lm_modes_j₁j₂)
	m_valid = m_range(lm_modes_j₁j₂)
	β_valid = intersect(0:1,vectorinds(j₁)) # no loop over this VSH
	γ_valid = vectorinds(j₂)

	m_compute = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? m_valid : (0:maximum(m_valid))
	m_symmetry = (sign(minimum(m_valid)) == sign(maximum(m_valid))) ? (0:-1) : (minimum(m_valid):-1)

	j₁norm = √((2j₁+1)/4π)

	@inbounds for m in m_compute

		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for m₁ in -1:1
			m₂ = m - m₁
			abs(m₂) > j₂ && continue

			CG_j₁m₁_j₂m₂_lm!(j₁,m₁,j₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

			for (ind,l) in enumerate(lrange_m)

				l_ind = (ind - 1) + first_l_ind # l's increase faster than m

				Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ = CG[l]
				iszero(Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂) && continue

				for γ in γ_valid, β in β_valid
					β != abs(m₁) && continue
					Yj₁j₂n₁n₂[β,abs(γ),l_ind] += j₁norm * Cˡᵐⱼ₁ₘ₁ⱼ₂ₘ₂ * Yj₂n₂[m₂,γ]
				end
			end
		end
	end

	@inbounds for m in m_symmetry
		
		lrange_m = l_range(lm_modes_j₁j₂,m)
		first_l_ind = modeindex(lm_modes,(first(lrange_m),m))

		for (ind,l) in enumerate(lrange_m)
			l_ind = (ind - 1) + first_l_ind # l's are stored contiguously
			l₋mind = modeindex(lm_modes,(l,-m))
			
			for γ in axes(Yj₁j₂n₁n₂,2), β in axes(Yj₁j₂n₁n₂,1)
				
				# In this case we may use the conjugation relations
				# Yʲ¹ʲ²ₗ₋ₘ_βγ = (-1)^(j₁+j₂+l+m+β+γ) conj(Yʲ¹ʲ²ₗₘ_-β-γ)

				Yj₁j₂n₁n₂[β,γ,l_ind] = (-1)^(j₁+j₂+l+m+β+γ) * conj(Yj₁j₂n₁n₂[β,γ,l₋mind])
			end
		end
	end

	# Specifically for m=0 all the components are either real or purely imaginary
	if 0 in m_valid
		@inbounds for l in l_range(lm_modes_j₁j₂,0)
			l_ind = modeindex(lm_modes,(l,0))
			if isodd(j₁+j₂+l)
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,0,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,1,l_ind]),0)
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,1,l_ind]))
			else
				Yj₁j₂n₁n₂[0,0,l_ind] = Complex(real(Yj₁j₂n₁n₂[0,0,l_ind]),0)
				Yj₁j₂n₁n₂[0,1,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[0,1,l_ind]))
				Yj₁j₂n₁n₂[1,0,l_ind] = Complex(0,imag(Yj₁j₂n₁n₂[1,0,l_ind]))
				Yj₁j₂n₁n₂[1,1,l_ind] = Complex(real(Yj₁j₂n₁n₂[1,1,l_ind]),0)
			end
		end
	end

	return Yj₁j₂n₁n₂
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

	Libdl.dlclose(lib)

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
function CG_j₁m₁_j₂m₂_lm(ℓ₁,m₁,ℓ₂,m=0;wig3j_fn_ptr=nothing)
	m₂ = m-m₁
	lmin = max(abs(ℓ₁-ℓ₂),abs(m))
	lmax = ℓ₁ + ℓ₂
	w3j = Wigner3j(ℓ₁,ℓ₂,m₁,m₂;wig3j_fn_ptr=wig3j_fn_ptr)
	CG = OffsetArray(w3j[1:(lmax-lmin+1)],lmin:lmax)
	CG_j₁m₁_j₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)
	return CG
end

function CG_j₁m₁_j₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG;wig3j_fn_ptr=nothing)
	m₂ = m-m₁
	w3j = Wigner3j(ℓ₁,ℓ₂,m₁,m₂;wig3j_fn_ptr=wig3j_fn_ptr)
	CG_j₁m₁_j₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)
	return CG
end

function CG_j₁m₁_j₂m₂_lm!(ℓ₁,m₁,ℓ₂,m,CG,w3j;wig3j_fn_ptr=nothing)
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

