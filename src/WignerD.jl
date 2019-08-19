module WignerD

using OffsetArrays, WignerSymbols, LinearAlgebra,Libdl
using PointsOnASphere,SphericalHarmonicModes
using SphericalHarmonics
import SphericalHarmonicModes: modeindex, s_valid_range, t_valid_range

export Ylmn,Ylmatrix,Ylmatrix!,djmatrix!,
djmn,djmatrix,BiPoSH_s0,BiPoSH,BiPoSH!,BSH,Jy_eigen,
st,ts,modes,modeindex,SphericalHarmonic,SphericalHarmonic!,st,ts

##########################################################################
# Wigner d matrix
##########################################################################

X(j,n) = sqrt((j+n)*(j-n+1))

function coeffi(j)
	N = 2j+1
	A = zeros(ComplexF64,N,N)
	coeffi!(j,A)
end

function coeffi!(j,A::Matrix{ComplexF64})

	N = 2j+1
	Av = @view A[1:N,1:N]

	if iszero(j)
		Av[1,1] = zero(ComplexF64)
		return Hermitian(Av)
	end

	A[1,2]=-X(j,-j+1)/2im
    A[N,N-1]=X(j,-j+1)/2im

    @inbounds for i in 2:N-1
	    A[i,i+1]=-X(j,-j+i)/2im
	    A[i,i-1]=X(j,j-i+2)/2im
	end

	return Hermitian(A)
end

function Jy_eigen(j)
	A = coeffi(j)
	λ,v = eigen(A)
	# We know that the eigenvalues of Jy are m ∈ -j:j, so we can round λ to integers and gain accuracy
	λ = round.(λ)
	#sort the array
	if issorted(λ)
		v = OffsetArray(collect(transpose(v)),-j:j,-j:j)
		λ = OffsetArray(λ,-j:j)
	else
		p = sortperm(λ)
		v = OffsetArray(collect(transpose(v[:,p])),-j:j,-j:j)
		λ = OffsetArray(λ[p],-j:j)
	end
	return λ,v
end

function Jy_eigen!(j,A)
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
	return λ,v
end

function djmatrix_fill!(dj,j,θ,m_range,n_range,λ,v)

	# check if symmetry conditions allow the index to be evaluated
	inds_covered = falses(m_range,n_range)

	@inbounds for (m,n) in Iterators.product(m_range,n_range)

		inds_covered[m,n] && continue

		dj_m_n = zero(ComplexF64)
		dj_m_n_πmθ = zero(ComplexF64)
		dj_n_m = zero(ComplexF64)

		@inbounds for μ in axes(λ,1)
			dj_m_n += cis(-λ[μ]*θ) * v[μ,m] * conj(v[μ,n])
			if m != n
				dj_n_m += cis(-λ[μ]*(-θ)) * v[μ,m] * conj(v[μ,n])
			end
			
			dj_m_n_πmθ += cis(-λ[μ]*(π-θ)) * v[μ,m] * conj(v[μ,n])
			
		end

		dj[m,n] = real(dj_m_n)
		inds_covered[m,n] = true
		if !iszero(m) && -m in m_range
			dj[-m,n] = real(dj_m_n_πmθ)*(-1)^(j+n)
			inds_covered[-m,n] = true
		end

		if !iszero(n) && -n in n_range
			dj[m,-n] = real(dj_m_n_πmθ)*(-1)^(j+m)
			inds_covered[m,-n] = true
		end

		if !(iszero(m) && iszero(n)) && -m in n_range && -n in m_range
			dj[-n,-m] = real(dj_m_n)
			inds_covered[-n,-m] = true
		end

		if  !iszero(n) && m !=n && -n in n_range && -m in m_range
			dj[-m,-n] = (-1)^(n+m) * real(dj_m_n)
			inds_covered[-m,-n] = true
		end

		# transpose
		if m != n && m in n_range && n in m_range
			dj[n,m] = real(dj_n_m)
			inds_covered[n,m] = true
		end
	end

	return dj
end

read_or_compute_eigen(j,::Nothing,::Nothing) = Jy_eigen(j)
read_or_compute_eigen(j,::Nothing,v) = (Float64(-j):Float64(j),v)
read_or_compute_eigen(j,λ,v) = (λ,v)

read_or_compute_eigen!(j,A,::Nothing,::Nothing) = Jy_eigen!(j,A)
read_or_compute_eigen!(j,A,::Nothing,v) = read_or_compute_eigen(j,v)
read_or_compute_eigen!(j,A,λ,v) = (λ,v)

struct djindices end
struct GSHindices end
struct OSHindices end

function get_m_n_ranges(j,::djindices;kwargs...)
	m_range=get(kwargs,:m_range,-j:j)
	n_range=get(kwargs,:n_range,-j:j)
	return m_range,n_range
end

function get_m_n_ranges(j,::GSHindices;kwargs...)
	m_range=get(kwargs,:m_range,-j:j)
	n_range=get(kwargs,:n_range,-1:1)
	return m_range,n_range
end

function get_m_n_ranges(j,::OSHindices;kwargs...)
	m_range=get(kwargs,:m_range,-j:j)
	n_range=0:0
	return m_range,n_range
end

# Default to full range
get_m_n_ranges(j;kwargs...) = get_m_n_ranges(j,djindices();kwargs...)

function djmatrix!(dj,j,θ::Real;kwargs...)

	λ = get(kwargs,:λ,nothing)
	v = get(kwargs,:v,nothing)
	m_range,n_range = get_m_n_ranges(j;kwargs...)

	λ,v = read_or_compute_eigen(j,λ,v)

	djmatrix_fill!(dj,j,θ,m_range,n_range,λ,v)
end

function djmatrix!(dj,j,θ::Real,A::Matrix{ComplexF64};kwargs...)

	λ = get(kwargs,:λ,nothing)
	v = get(kwargs,:v,nothing)
	m_range,n_range = get_m_n_ranges(j;kwargs...)

	λ,v = read_or_compute_eigen!(j,A,λ,v)

	djmatrix_fill!(dj,j,θ,m_range,n_range,λ,v)
end

function djmatrix(j,θ;kwargs...)
	m_range,n_range = get_m_n_ranges(j;kwargs...)
	dj = zeros(m_range,n_range)
	djmatrix!(dj,j,θ;m_range=m_range,n_range=n_range,kwargs...)
end

djmatrix(j,x::SphericalPoint;kwargs...) = djmatrix(j,x.θ;kwargs...)
djmatrix(j,m,n,θ::Real;kwargs...) = djmatrix(j,θ,m_range=m:m,n_range=n:n;kwargs...)
djmatrix(j,m,n,x::SphericalPoint;kwargs...) = djmatrix(j,x.θ,m_range=m:m,n_range=n:n;kwargs...)

djmatrix!(dj::AbstractMatrix{<:Real},j,x::SphericalPoint;kwargs...) = djmatrix!(dj,j,x.θ;kwargs...)
djmatrix!(dj::AbstractMatrix{<:Real},j,m,n,θ::Real;kwargs...) = djmatrix!(dj,j,θ,m_range=m:m,n_range=n:n;kwargs...)
djmatrix!(dj::AbstractMatrix{<:Real},j,m,n,x::SphericalPoint;kwargs...) = djmatrix!(dj,j,x.θ,m_range=m:m,n_range=n:n;kwargs...)

##########################################################################
# Generalized spherical harmonics
# Resort to spherical harmonics if n=0
##########################################################################

abstract type AbstractSH end
struct GSH <: AbstractSH end # Generalized SH
struct OSH <: AbstractSH end # Ordinary SH

# Use the generalized spherical harmonics by default
Ylmatrix(args...;kwargs...) = Ylmatrix(GSH(),args...;kwargs...)
Ylmatrix!(args...;kwargs...) = Ylmatrix!(GSH(),args...;kwargs...)

function Ylmatrix(::GSH,l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)

	n_range = last(get_m_n_ranges(l,GSHindices();kwargs...))

	dj_θ = djmatrix(l,θ;kwargs...,n_range=n_range)
	Y = zeros(ComplexF64,axes(dj_θ)...)
	Ylmatrix!(Y,dj_θ,l,(θ,ϕ);n_range=n_range,kwargs...,compute_d_matrix=false)
end

function Ylmatrix(::GSH,dj_θ::AbstractMatrix{<:Real},l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)

	n_range = last(get_m_n_ranges(l,GSHindices();kwargs...))
	m_range = axes(dj_θ,1)

	Y = zeros(ComplexF64,m_range,n_range)
	Ylmatrix!(Y,dj_θ,l,(θ,ϕ);compute_d_matrix=false,n_range=n_range,kwargs...)
end

function Ylmatrix(::OSH,l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)
	m_range,n_range = get_m_n_ranges(l,OSHindices();kwargs...)
	Y = zeros(ComplexF64,m_range,n_range)
	Ylmatrix!(OSH(),Y,l,(θ,ϕ);kwargs...)
end

function Ylmatrix!(::GSH,Y::AbstractMatrix{<:Complex},l::Integer,
	(θ,ϕ)::Tuple{<:Real,<:Real},args...;kwargs...)

	m_range,n_range = get_m_n_ranges(l,GSHindices();kwargs...)

	dj_θ = zeros(m_range,n_range)

	Ylmatrix!(Y,dj_θ,l,(θ,ϕ),args...;n_range=n_range,kwargs...,compute_d_matrix=true)
end

function Ylmatrix!(::GSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},args...;kwargs...)

	m_range,n_range = get_m_n_ranges(l,GSHindices();kwargs...)

	if get(kwargs,:compute_d_matrix,false):: Bool
		djmatrix!(dj_θ,l,θ,args...;kwargs...,n_range=n_range)
	end

	@inbounds for (m,n) in Iterators.product(m_range,n_range)
		Y[m,n] = √((2l+1)/4π) * dj_θ[m,n] * cis(m*ϕ)
	end
	return Y
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)
	YSH = SphericalHarmonics.allocate_y(l)
	Ylmatrix!(OSH(),Y,l,(θ,ϕ),YSH;kwargs...,compute_Ylm=true,compute_Pl=true)
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},
	YSH::AbstractVector{<:Complex};kwargs...)

	P = get(kwargs,:compute_Pl,false) ? compute_p(l,cos(θ)) : nothing
	
	Ylmatrix!(OSH(),Y,l,(θ,ϕ),P,YSH;kwargs...)
end

function YSH_loop!(Y::AbstractMatrix{<:Complex},YSH::AbstractVector{<:Complex},
	l::Integer,m_range::AbstractUnitRange)

	@inbounds for m in m_range
		lmind = index_y(l,m)
		Y[m,0] = YSH[lmind]
	end
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},
	P::Vector{<:Real},YSH::AbstractVector{<:Complex};kwargs...)

	m_range = first(get_m_n_ranges(l,OSHindices();kwargs...))

	if get(kwargs,:compute_Pl,false)
		coeff = SphericalHarmonics.compute_coefficients(l)
		compute_p!(l,cos(θ),coeff,P)
	end

	if get(kwargs,:compute_Ylm,false)
		compute_y!(l,cos(θ),ϕ,P,YSH)
	end

	YSH_loop!(Y,YSH,l,m_range)

	return Y
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},
	P::Nothing,YSH::AbstractVector{<:Complex};kwargs...)

	m_range = first(get_m_n_ranges(l,OSHindices();kwargs...))
	if get(kwargs,:compute_Ylm,false)
		compute_y!(l,cos(θ),ϕ,YSH)
	end
	YSH_loop!(Y,YSH,l,m_range)
	return Y
end

Ylmatrix(T::GSH,l::Integer,m::Integer,n::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = 
	Ylmatrix(T,l,(θ,ϕ);kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix(T::OSH,l::Integer,m::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = 
	Ylmatrix(T,l,(θ,ϕ);kwargs...,m_range=m:m)

Ylmatrix(T::GSH,l::Integer,m::Integer,n::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,l,(x.θ,x.ϕ);kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix(T::OSH,l::Integer,m::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,l,(x.θ,x.ϕ);kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix(T::AbstractSH,l::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,l,(x.θ,x.ϕ);kwargs...)

Ylmatrix(T::GSH,dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,n::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = 
	Ylmatrix(T,dj_θ,l,(θ,ϕ);kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix(T::OSH,dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...) = 
	Ylmatrix(T,dj_θ,l,(θ,ϕ);kwargs...,m_range=m:m)

Ylmatrix(T::GSH,dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,n::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,dj_θ,l,(x.θ,x.ϕ);kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix(T::OSH,dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,dj_θ,l,(x.θ,x.ϕ);kwargs...,m_range=m:m)

Ylmatrix(T::AbstractSH,dj_θ::AbstractMatrix{<:Real},l::Integer,x::SphericalPoint;kwargs...) = 
	Ylmatrix(T,dj_θ,l,(x.θ,x.ϕ);kwargs...)

Ylmatrix!(T::GSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,n::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},args...;kwargs...) = 
	Ylmatrix!(Y,dj_θ,l,(θ,ϕ),args...;kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix!(T::OSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,n::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},args...;kwargs...) = 
	Ylmatrix!(Y,dj_θ,l,(θ,ϕ),args...;kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix!(T::GSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,n::Integer,x::SphericalPoint,args...;kwargs...) = 
	Ylmatrix!(T,Y,dj_θ,l,(x.θ,x.ϕ),args...;kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix!(T::OSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},l::Integer,m::Integer,x::SphericalPoint,args...;kwargs...) = 
	Ylmatrix!(T,Y,dj_θ,l,(x.θ,x.ϕ),args...;kwargs...,m_range=m:m)

Ylmatrix!(T::AbstractSH,Y::AbstractMatrix{<:Complex},dj_θ::AbstractMatrix{<:Real},l::Integer,x::SphericalPoint,args...;kwargs...) = 
	Ylmatrix!(T,Y,dj_θ,l,(x.θ,x.ϕ),args...;kwargs...)

Ylmatrix!(T::GSH,Y::AbstractMatrix{<:Complex},l::Integer,m::Integer,n::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},args...;kwargs...) = 
	Ylmatrix!(T,Y,l,(θ,ϕ),args...;kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix!(T::OSH,Y::AbstractMatrix{<:Complex},l::Integer,m::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},args...;kwargs...) = 
	Ylmatrix!(T,Y,l,(θ,ϕ),args...;kwargs...,m_range=m:m)

Ylmatrix!(T::GSH,Y::AbstractMatrix{<:Complex},l::Integer,m::Integer,n::Integer,x::SphericalPoint,args...;kwargs...) = 
	Ylmatrix!(T,Y,l,(x.θ,x.ϕ),args...;kwargs...,m_range=m:m,n_range=n:n)

Ylmatrix!(T::OSH,Y::AbstractMatrix{<:Complex},l::Integer,m::Integer,x::SphericalPoint,args...;kwargs...) = 
	Ylmatrix!(T,Y,l,(x.θ,x.ϕ),args...;kwargs...,m_range=m:m)

Ylmatrix!(T::AbstractSH,Y::AbstractMatrix{<:Complex},l::Integer,x::SphericalPoint,args...;kwargs...) = 
	Ylmatrix!(T,Y,l,(x.θ,x.ϕ),args...;kwargs...)

Ylmatrix!(T::AbstractSH,Y::AbstractMatrix{<:Complex},::Nothing,args...;kwargs...) = 
	Ylmatrix!(T,Y,args...;kwargs...)

##########################################################################
# Spherical harmonics
##########################################################################

function SphericalHarmonic(args...;kwargs...)
	Y = Ylmatrix(OSH(),args...;kwargs...)
	Y[:,0]
end

function SphericalHarmonic!(Y::AbstractMatrix{<:Complex},args...;kwargs...)
	Y = Ylmatrix!(OSH(),Y,args...;kwargs...)
	Y[:,0]
end

function SphericalHarmonic!(Y::AbstractVector{<:Complex},args...;kwargs...)
	Y2D = reshape(Y,axes(Y,1),0:0)
	SphericalHarmonic!(Y2D,args...;kwargs...)
	Y2D[:,0]
end

##########################################################################
# Bipolar Spherical harmonics
##########################################################################

##################################################################################################

# Convenience function to convert an integer to a UnitRange to be used as an array axis
to_unitrange(a::Integer) = a:a
to_unitrange(a::AbstractUnitRange) = a

# Only t=0
function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))
	# only t=0
	if iszero(length(Y_ℓ₁)) 
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	if iszero(length(Y_ℓ₂))
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂) ::Integer

	Y_BSH = zeros(ComplexF64,s:s,β:β,γ:γ)

	@inbounds for m in -m_max:m_max
		Y_BSH[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing,
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))
	# only t=0

	if iszero(length(Y_ℓ₁)) 
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	if iszero(length(Y_ℓ₂))
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	m_max = min(ℓ₁,ℓ₂)

	s_valid_range = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_range,s_valid_range)

	Y_BSH = zeros(ComplexF64,s_intersection,β:β,γ:γ)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds for m in -m_max:m_max
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Y_BSH,1),axes(CG,1))
		
		@inbounds for s in s_intersection
			Y_BSH[s,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))

	# only t=0
	if iszero(length(Y_ℓ₁))
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	if iszero(length(Y_ℓ₂)) 
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂)

	Y_BSH = zeros(ComplexF64,s:s,-1:1,-1:1)

	@inbounds for (s,β,γ) in Iterators.product(axes(Y_BSH)...),m in -m_max:m_max
		Y_BSH[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
	end

	return Y_BSH
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing,
	Y_ℓ₁=zeros(0:-1,0:-1),Y_ℓ₂=zeros(0:-1,0:-1))

	if iszero(length(Y_ℓ₁))
		Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	if iszero(length(Y_ℓ₂)) 
		Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	m_max = min(ℓ₁,ℓ₂)

	s_valid_range = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_valid_range,s_range)

	Y_BSH = zeros(ComplexF64,s_intersection,-1:1,-1:1)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds  for m in -m_max:m_max
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Y_BSH,1),axes(CG,1))

		@inbounds for (s,β,γ) in Iterators.product(s_intersection,axes(Y_BSH)[2:3]...)
			Y_BSH[s,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[-m,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

BiPoSH_s0(ℓ₁,ℓ₂,s,β::Integer,γ::Integer,
	x::SphericalPoint,x2::SphericalPoint;kwargs...) = BiPoSH_s0(ℓ₁,ℓ₂,s,β,γ,(x.θ,x.ϕ),(x2.θ,x2.ϕ);kwargs...)

BiPoSH_s0(ℓ₁,ℓ₂,s,
	x::SphericalPoint,x2::SphericalPoint;kwargs...) = BiPoSH_s0(ℓ₁,ℓ₂,s,(x.θ,x.ϕ),(x2.θ,x2.ϕ);kwargs...)

# Any t

struct BSH{T} <: AbstractArray{ComplexF64,3}
	modes :: T
	parent :: OffsetArray{ComplexF64,3,Array{ComplexF64,3}}
end

function BSH{T}(smin::Integer,smax::Integer,tmin::Integer,tmax::Integer,
	β_range::Union{Integer,AbstractUnitRange},
	γ_range::Union{Integer,AbstractUnitRange},
	arr::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}) where {T<:SHModeRange}

	st_iterator = T(smin,smax,tmin,tmax)
	β_range,γ_range = to_unitrange.((β_range,γ_range))
	BSH{T}(st_iterator,arr)
end

function BSH{T}(smin::Integer,smax::Integer,tmin::Integer,tmax::Integer,
	β_range::Union{Integer,AbstractUnitRange},
	γ_range::Union{Integer,AbstractUnitRange},
	arr::AbstractArray{ComplexF64,3}) where {T<:SHModeRange}

	BSH{T}(smin,smax,tmin,tmax,β_range,γ_range,
		OffsetArray(arr,axes(arr,1),β_range,γ_range))
end

function BSH{T}(smin::Integer,smax::Integer,tmin::Integer,tmax::Integer,
	β_range::Union{Integer,AbstractUnitRange}=-1:1,
	γ_range::Union{Integer,AbstractUnitRange}=-1:1) where {T<:SHModeRange}

	st_iterator = T(smin,smax,tmin,tmax)
	BSH{T}(st_iterator,
		zeros(ComplexF64,length(st_iterator),β_range,γ_range))
end

BSH{T}(s_range::AbstractUnitRange,t_range::AbstractUnitRange,args...) where {T<:SHModeRange} = 
	BSH{T}(minimum(s_range),maximum(s_range),minimum(t_range),maximum(t_range),args...)

BSH{T}(s_range::AbstractUnitRange,t::Integer,args...) where {T<:SHModeRange} = 
	BSH{T}(minimum(s_range),maximum(s_range),t,t,args...)

BSH{T}(s::Integer,t_range::AbstractUnitRange,args...) where {T<:SHModeRange} = 
	BSH{T}(s,s,minimum(t_range),maximum(t_range),args...)

BSH{T}(s::Integer,t::Integer,args...) where {T<:SHModeRange} = BSH{T}(s,s,t,t,args...)

Base.similar(b::BSH{T}) where {T<:SHModeRange} = BSH{T}(s_range(b),t_range(b),axes(parent(b))[2:3]...)
Base.copy(b::BSH{T}) where {T<:SHModeRange} = BSH{T}(modes(b),copy(parent(b)))

modes(b::BSH) = b.modes
modeindex(b::BSH,s::Integer,t::Integer) = modeindex(modes(b),s,t)
modeindex(b::BSH,::Colon,t::Integer) = 
	[modeindex(modes(b),s,t)  for s in s_valid_range(b,t)]

modeindex(b::BSH,s::Integer,::Colon) = 
	[modeindex(modes(b),s,t)  for t in t_valid_range(b,s)]

modeindex(b::BSH,::Colon,::Colon) = axes(parent(b),1)

s_range(b::BSH) = modes(b).smin:modes(b).smax
t_range(b::BSH) = modes(b).tmin:modes(b).tmax

Base.parent(b::BSH) = b.parent

Base.size(b::BSH) = size(parent(b))
Base.size(b::BSH,d) = size(parent(b),d)
Base.axes(b::BSH) = axes(parent(b))
Base.axes(b::BSH,d) = axes(parent(b),d)

Base.getindex(b::BSH,s,t,args...) = parent(b)[modeindex(b,s,t),args...]
Base.setindex!(b::BSH,x,s,t,args...) = parent(b)[modeindex(b,s,t),args...] = x

Base.fill!(a::BSH,x) = fill!(parent(a),x)

s_valid_range(b::BSH,t::Integer) = s_valid_range(modes(b),t)
t_valid_range(b::BSH,s::Integer) = t_valid_range(modes(b),s)

function Base.show(io::IO, b::BSH)
    compact = get(io, :compact, false)

    smin = b.modes.smin
    smax = b.modes.smax
    tmin = b.modes.tmin
    tmax = b.modes.tmax
    β_range = convert(UnitRange{Int64},axes(parent(b),2))
    γ_range = convert(UnitRange{Int64},axes(parent(b),3))

    println("s=$smin:$smax and t=$tmin:$tmax, vector indices β=$β_range and γ=$γ_range")
    display(parent(b))
end

function Base.show(io::IO, ::MIME"text/plain", b::BSH)
	println("Bipolar spherical harmonic in the Phinney-Burridge basis")
    show(io, b)  
end

function BiPoSH(ℓ₁,ℓ₂,s::Integer,t::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	β::Integer,γ::Integer)

	Y_BSH = BiPoSH(ℓ₁,ℓ₂,s,t,(θ₁,ϕ₁),(θ₂,ϕ₂),β:β,γ:γ)
	Y_BSH[s,t,β,γ]
end

function BiPoSH(ℓ₁,ℓ₂,s::Integer,t::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	β_range::AbstractUnitRange=-1:1,
	γ_range::AbstractUnitRange=-1:1)
	
	Y_ℓ₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β_range)
	Y_ℓ₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ_range)
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied for ℓ₁=$ℓ₁, ℓ₂=$ℓ₂ and s=$s")
	@assert(abs(t)<=s,"abs(t)<=s not satisfied for t=$t and s=$s")

	Y_BSH = BSH{st}(s:s,t:t,β_range,γ_range)

	for γ in γ_range,β in β_range
		for m in -ℓ₁:ℓ₁
			n = t - m
			if abs(n)>ℓ₂
				continue
			end
			Y_BSH[s,t,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,n,s,t)*Y_ℓ₁[m,β]*Y_ℓ₂[n,γ]
		end
	end

	return Y_BSH
end

function BiPoSH(ℓ₁,ℓ₂,s_range::AbstractUnitRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real};
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	t::Union{Integer,AbstractUnitRange}=-last(s_range):last(s_range),
	kwargs...)

	β,γ,t = to_unitrange.((β,γ,t))

	Y_ℓ₁ = zeros(ComplexF64,-ℓ₁:ℓ₁,β)
	Y_ℓ₂ = zeros(ComplexF64,-ℓ₂:ℓ₂,γ)

	BiPoSH!(ℓ₁,ℓ₂,s_range,(θ₁,ϕ₁),(θ₂,ϕ₂),Y_ℓ₁,Y_ℓ₂;
		β=β,γ=γ,t=t,compute_Yℓ₁=true,compute_Yℓ₂=true,kwargs...)
end

function BiPoSH!(ℓ₁,ℓ₂,s_range::AbstractUnitRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Y_ℓ₁,Y_ℓ₂;
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	t::Union{Integer,AbstractUnitRange}=-last(s_range):last(s_range),
	wig3j_fn_ptr=nothing,compute_Yℓ₁=false,compute_Yℓ₂=false,kwargs...)

	β,γ,t = to_unitrange.((β,γ,t))

	if compute_Yℓ₁
		dℓ₁ = zeros(-ℓ₁:ℓ₁,β)
		Ylmatrix!(Y_ℓ₁,dℓ₁,ℓ₁,(θ₁,ϕ₁),n_range=β,compute_d_matrix=true,kwargs...)
	end

	if compute_Yℓ₂
		dℓ₂ = ((ℓ₁ == ℓ₂) && (θ₁ == θ₂) && compute_Yℓ₁) ? dℓ₁ : zeros(-ℓ₂:ℓ₂,γ)
		Ylmatrix!(Y_ℓ₂,dℓ₂,ℓ₂,(θ₂,ϕ₂),n_range=γ,
			compute_d_matrix = compute_Yℓ₁ && (dℓ₁ !== dℓ₂),kwargs...)
	end

	BiPoSH_compute!(ℓ₁,ℓ₂,s_range,(θ₁,ϕ₁),(θ₂,ϕ₂),Y_ℓ₁,Y_ℓ₂,β,γ,t;
		wig3j_fn_ptr=wig3j_fn_ptr,compute_Yℓ₁=false,compute_Yℓ₂=false)
end

function BiPoSH!(ℓ₁,ℓ₂,s_range::AbstractUnitRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Y_ℓ₁,Y_ℓ₂,dℓ₁,dℓ₂;
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	t::Union{Integer,AbstractUnitRange}=-last(s_range):last(s_range),
	wig3j_fn_ptr=nothing,
	compute_dℓ₁=false,compute_dℓ₂=false,
	compute_Yℓ₁=false,compute_Yℓ₂=false,kwargs...)

	β,γ,t = to_unitrange.((β,γ,t))

	if compute_Yℓ₁
		Ylmatrix!(Y_ℓ₁,dℓ₁,ℓ₁,(θ₁,ϕ₁),n_range=β,compute_d_matrix=compute_dℓ₁,kwargs...)
	end
	if compute_Yℓ₂
		Ylmatrix!(Y_ℓ₂,dℓ₂,ℓ₂,(θ₂,ϕ₂),n_range=γ,
			compute_d_matrix=(compute_dℓ₂ && (dℓ₁ !== dℓ₂)),kwargs...)
	end

	BiPoSH_compute!(ℓ₁,ℓ₂,s_range,(θ₁,ϕ₁),(θ₂,ϕ₂),Y_ℓ₁,Y_ℓ₂,β,γ,t;
		wig3j_fn_ptr=wig3j_fn_ptr,compute_Yℓ₁=false,compute_Yℓ₂=false)
end

function BiPoSH_compute!(ℓ₁,ℓ₂,s_range::AbstractUnitRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Y_ℓ₁::AbstractArray,Y_ℓ₂::AbstractArray,
	β_range::AbstractUnitRange=-1:1,
	γ_range::AbstractUnitRange=-1:1,
	t_range::AbstractUnitRange=-last(s_range):last(s_range),
	dℓ₁=nothing,dℓ₂=nothing;
	compute_dℓ₁=false,compute_dℓ₂=false,
	compute_Yℓ₁=true,compute_Yℓ₂=true,
	wig3j_fn_ptr=nothing)

	if compute_Yℓ₁
		Ylmatrix!(Y_ℓ₁,dℓ₁,ℓ₁,(θ₁,ϕ₁),n_range=β_range,
			compute_d_matrix= !isnothing(dℓ₁) && compute_d_ℓ₁)
	end
	if compute_Yℓ₂
		Ylmatrix!(Y_ℓ₂,dℓ₂,ℓ₂,(θ₂,ϕ₂),n_range=γ_range,
			compute_d_matrix= !isnothing(dℓ₂) && compute_d_ℓ₂)
	end

	s_ℓ₁ℓ₂ = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_range = intersect(s_ℓ₁ℓ₂,s_range)
	t_range = intersect(-maximum(s_range):maximum(s_range),t_range)

	Y_BSH = BSH{st}(s_range,t_range,β_range,γ_range)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds for γ in γ_range,β in β_range, t in t_range, m in -ℓ₁:ℓ₁
		
		n = t - m
		if abs(n) > ℓ₂
			continue
		end
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂,t;wig3j_fn_ptr=wig3j_fn_ptr)

		@inbounds for s in s_valid_range(Y_BSH,t)
			Y_BSH[s,t,β,γ] += CG[s]*Y_ℓ₁[m,β]*Y_ℓ₂[n,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Y_BSH
end

BiPoSH(ℓ₁,ℓ₂,s::Integer,t::Integer,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH(ℓ₁,ℓ₂,s,t,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

BiPoSH(ℓ₁,ℓ₂,s_range::AbstractUnitRange,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH(ℓ₁,ℓ₂,s_range,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

BiPoSH!(ℓ₁,ℓ₂,s_range::AbstractUnitRange,x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH!(ℓ₁,ℓ₂,s_range,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

##################################################################################################

function Wigner3j(j2,j3,m2,m3;wig3j_fn_ptr=nothing)
	
	m2,m3 = Int32(m2),Int32(m3)
	m1 = Int32(-(m2 + m3))

	j2,j3 = Int32(j2),Int32(j3)
	len = Int32(j2+j3+1)

	exitstatus = zero(Int32)

	w3j = zeros(Float64,len)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
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

	if !isnothing(lib)
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

	if isnothing(wig3j_fn_ptr)
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

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end
end

function CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂,t=0;wig3j_fn_ptr=nothing)
	n = t-m
	smin = max(abs(ℓ₁-ℓ₂),abs(t))
	smax = ℓ₁ + ℓ₂
	w = Wigner3j(ℓ₁,ℓ₂,m,n;wig3j_fn_ptr=wig3j_fn_ptr)
	CG = OffsetArray(w[1:(smax-smin+1)],smin:smax)
	@inbounds for s in axes(CG,1)
		CG[s] *= √(2s+1)*(-1)^(ℓ₁-ℓ₂)
	end
	return CG
end

include("./precompile.jl")

end

