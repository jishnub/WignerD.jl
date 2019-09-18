module WignerD

using OffsetArrays, WignerSymbols, LinearAlgebra,Libdl
using PointsOnASphere,SphericalHarmonicModes
using SphericalHarmonics
import SphericalHarmonicModes: modeindex, s_valid_range, t_valid_range,
s_range,t_range

export Ylmn,Ylmatrix,Ylmatrix!,djmatrix!,
djmn,djmatrix,BiPoSH_s0,BiPoSH,BiPoSH_n1n2_n2n1,BiPoSH!,BSH,Jy_eigen,
st,ts,s′s,modes,modeindex,SphericalHarmonic,SphericalHarmonic!,st,ts,
OSH,GSH

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

	if get(kwargs,:compute_d_matrix,true):: Bool
		djmatrix!(dj_θ,l,θ,args...;kwargs...,n_range=n_range)
	end

	@inbounds for (m,n) in Iterators.product(m_range,n_range)
		Y[m,n] = √((2l+1)/4π) * dj_θ[m,n] * cis(m*ϕ)
	end
	return Y
end

function YSH_fill!(Y::AbstractMatrix{<:Complex},YSH::AbstractVector{<:Complex},
	l::Integer,m_range::AbstractUnitRange=axes(Y,1))

	@assert(!isempty(intersect(m_range,axes(Y,1))),
		"Y is not large enough to contain m range $m_range."*
		" Axes of Y is $(axes(Y)) and l=$l, m=$m_range")

	@assert(length(YSH)>=SphericalHarmonics.sizeY(l),
		"YSH does not contain all (l,m)."*
		"Axes of YSH is $(axes(YSH)) and l=$l, m=$m_range")

	@inbounds for m in m_range
		Y[m,0] = YSH[index_y(l,m)]
	end
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};kwargs...)

	YSH = SphericalHarmonics.allocate_y(l)
	Ylmatrix!(OSH(),Y,YSH,l,(θ,ϕ);kwargs...,compute_Ylm=true,compute_Pl=true)
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	YSH::AbstractVector{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real};
	kwargs...)

	P = get(kwargs,:compute_Pl,false) ? compute_p(l,cos(θ)) : nothing
	
	Ylmatrix!(OSH(),Y,YSH,l,(θ,ϕ),P;kwargs...)
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	YSH::AbstractVector{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},
	P::Vector{<:Real},Pcoeff=nothing;kwargs...)

	m_range = get_m_n_ranges(l,OSHindices();kwargs...) |> first

	if get(kwargs,:compute_Pl,false)
		if isnothing(Pcoeff)
			Pcoeff = SphericalHarmonics.compute_coefficients(l)
		end
		compute_p!(l,cos(θ),Pcoeff,P)
	end

	if get(kwargs,:compute_Ylm,false)
		compute_y!(l,ϕ,P,YSH)
	end

	YSH_fill!(Y,YSH,l,m_range)

	return Y
end

function Ylmatrix!(::OSH,Y::AbstractMatrix{<:Complex},
	YSH::AbstractVector{<:Complex},
	l::Integer,(θ,ϕ)::Tuple{<:Real,<:Real},
	P::Nothing;kwargs...)

	m_range = first(get_m_n_ranges(l,OSHindices();kwargs...))
	if get(kwargs,:compute_Ylm,false)
		compute_y!(l,cos(θ),ϕ,YSH)
	end
	YSH_fill!(Y,YSH,l,m_range)
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
	Yℓ₁n₁=zeros(0:-1,0:-1),Yℓ₂n₂=zeros(0:-1,0:-1))
	# only t=0
	if iszero(length(Yℓ₁n₁)) 
		Yℓ₁n₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	if iszero(length(Yℓ₂n₂))
		Yℓ₂n₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂) ::Integer

	Yℓ₁ℓ₂n₁n₂ = zeros(ComplexF64,s:s,β:β,γ:γ)

	@inbounds for m in -m_max:m_max
		Yℓ₁ℓ₂n₁n₂[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Yℓ₁n₁[m,β]*Yℓ₂n₂[-m,γ]
	end

	return Yℓ₁ℓ₂n₁n₂
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,β::Integer,γ::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing,
	Yℓ₁n₁=zeros(0:-1,0:-1),Yℓ₂n₂=zeros(0:-1,0:-1))
	# only t=0

	if iszero(length(Yℓ₁n₁)) 
		Yℓ₁n₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β:β) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	if iszero(length(Yℓ₂n₂))
		Yℓ₂n₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ:γ) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end
	m_max = min(ℓ₁,ℓ₂)

	s_valid_range = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_range,s_valid_range)

	Yℓ₁ℓ₂n₁n₂ = zeros(ComplexF64,s_intersection,β:β,γ:γ)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds for m in -m_max:m_max
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Yℓ₁ℓ₂n₁n₂,1),axes(CG,1))
		
		@inbounds for s in s_intersection
			Yℓ₁ℓ₂n₁n₂[s,β,γ] += CG[s]*Yℓ₁n₁[m,β]*Yℓ₂n₂[-m,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Yℓ₁ℓ₂n₁n₂
end

function BiPoSH_s0(ℓ₁,ℓ₂,s::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};
	Yℓ₁n₁=zeros(0:-1,0:-1),Yℓ₂n₂=zeros(0:-1,0:-1))

	# only t=0
	if iszero(length(Yℓ₁n₁))
		Yℓ₁n₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	if iszero(length(Yℓ₂n₂)) 
		Yℓ₂n₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied")
	m_max = min(ℓ₁,ℓ₂)

	Yℓ₁ℓ₂n₁n₂ = zeros(ComplexF64,s:s,-1:1,-1:1)

	@inbounds for (s,β,γ) in Iterators.product(axes(Yℓ₁ℓ₂n₁n₂)...),m in -m_max:m_max
		Yℓ₁ℓ₂n₁n₂[s,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,-m,s,0)*Yℓ₁n₁[m,β]*Yℓ₂n₂[-m,γ]
	end

	return Yℓ₁ℓ₂n₁n₂
end

function BiPoSH_s0(ℓ₁,ℓ₂,s_range::AbstractRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real};wig3j_fn_ptr=nothing,
	Yℓ₁n₁=zeros(0:-1,0:-1),Yℓ₂n₂=zeros(0:-1,0:-1))

	if iszero(length(Yℓ₁n₁))
		Yℓ₁n₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	if iszero(length(Yℓ₂n₂)) 
		Yℓ₂n₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂)) :: OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
	end

	m_max = min(ℓ₁,ℓ₂)

	s_valid_range = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_intersection = intersect(s_valid_range,s_range)

	Yℓ₁ℓ₂n₁n₂ = zeros(ComplexF64,s_intersection,-1:1,-1:1)

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	@inbounds  for m in -m_max:m_max
		CG = CG_ℓ₁mℓ₂nst(ℓ₁,m,ℓ₂;wig3j_fn_ptr=wig3j_fn_ptr)

		s_intersection = intersect(axes(Yℓ₁ℓ₂n₁n₂,1),axes(CG,1))

		@inbounds for (s,β,γ) in Iterators.product(s_intersection,axes(Yℓ₁ℓ₂n₁n₂)[2:3]...)
			Yℓ₁ℓ₂n₁n₂[s,β,γ] += CG[s]*Yℓ₁n₁[m,β]*Yℓ₂n₂[-m,γ]
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
	end

	return Yℓ₁ℓ₂n₁n₂
end

BiPoSH_s0(ℓ₁,ℓ₂,s,β::Integer,γ::Integer,
	x::SphericalPoint,x2::SphericalPoint;kwargs...) = 
	BiPoSH_s0(ℓ₁,ℓ₂,s,β,γ,(x.θ,x.ϕ),(x2.θ,x2.ϕ);kwargs...)

BiPoSH_s0(ℓ₁,ℓ₂,s,x::SphericalPoint,x2::SphericalPoint;kwargs...) = 
	BiPoSH_s0(ℓ₁,ℓ₂,s,(x.θ,x.ϕ),(x2.θ,x2.ϕ);kwargs...)

# Any t

struct BSH{TSH<:SHModeRange,N,AA<:AbstractArray{ComplexF64,N}} <: AbstractArray{ComplexF64,N}
	modes :: TSH
	parent :: AA
end

function BSH(st_iterator::SHModeRange,
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1)

	β,γ = to_unitrange.((β,γ))
	BSH(st_iterator,zeros(ComplexF64,length(st_iterator),β,γ))
end

function BSH{T}(smin::Integer,smax::Integer,tmin::Integer,tmax::Integer,
	arr::AbstractArray{ComplexF64,3}) where {T<:SHModeRange}

	st_iterator = T(smin,smax,tmin,tmax)
	BSH(st_iterator,arr)
end

function BSH{T}(smin::Integer,smax::Integer,tmin::Integer,tmax::Integer;
	kwargs...) where {T<:SHModeRange}

	st_iterator = T(smin,smax,tmin,tmax)
	BSH(st_iterator;kwargs...)
end

BSH{T}(s_range::AbstractUnitRange,t_range::AbstractUnitRange;kwargs...) where {T<:SHModeRange} = 
	BSH{T}(minimum(s_range),maximum(s_range),minimum(t_range),maximum(t_range);kwargs...)

BSH{T}(s_range::AbstractUnitRange,t::Integer;kwargs...) where {T<:SHModeRange} = 
	BSH{T}(minimum(s_range),maximum(s_range),t,t;kwargs...)

BSH{T}(s::Integer,t_range::AbstractUnitRange;kwargs...) where {T<:SHModeRange} = 
	BSH{T}(s,s,minimum(t_range),maximum(t_range);kwargs...)

BSH{T}(s::Integer,t::Integer;kwargs...) where {T<:SHModeRange} = BSH{T}(s,s,t,t;kwargs...)

Base.similar(b::BSH{T}) where {T<:SHModeRange} = BSH{T}(s_range(b),t_range(b),axes(parent(b))[2:3]...)
Base.copy(b::BSH{T}) where {T<:SHModeRange} = BSH{T}(modes(b),copy(parent(b)))

modes(b::BSH) = b.modes

modeindex(b::BSH,s,t) = modeindex(modes(b),s,t)
modeindex(b::BSH,::Colon,::Colon) = axes(parent(b),1)

s_range(b::BSH) = s_range(modes(b))
t_range(b::BSH) = t_range(modes(b))

Base.parent(b::BSH) = b.parent

Base.size(b::BSH) = size(parent(b))
Base.size(b::BSH,d) = size(parent(b),d)
Base.axes(b::BSH) = axes(parent(b))
Base.axes(b::BSH,d) = axes(parent(b),d)

Base.view(b::BSH,args...) = BSH(modes(b),view(parent(b),args...))
Base.view(b::BSH{T,1},s,t) where {T<:SHModeRange} = 
	BSH(modes(b),view(parent(b),modeindex(b,s,t)))

function Base.getindex(b::BSH,s::Integer,t::Integer,args...)
	stind = modeindex(b,s,t)
	parent(b)[stind,args...]
end

# This method is necessary in case we are using 1D views for a constant t
Base.getindex(b::BSH{st,1},s::Integer) = parent(b)[s]
# This method is necessary in case we are using 1D views for a constant s
Base.getindex(b::BSH{ts,1},t::Integer) = parent(b)[t]

Base.getindex(b::BSH,::Colon,args...) = parent(b)[:,args...]

function Base.setindex!(b::BSH,x,s::Integer,t::Integer,args...)
	stind = modeindex(b,s,t)
	parent(b)[stind,args...] = x
end

Base.setindex!(b::BSH,x,::Colon,args...) = parent(b)[:,args...] = x

# This method is necessary in case we are using 1D views for a constant t
Base.setindex!(b::BSH{st,1},x,s::Integer) = parent(b)[s] = x
# This method is necessary in case we are using 1D views for a constant s
Base.setindex!(b::BSH{ts,1},x,t::Integer) = parent(b)[t] = x

Base.fill!(b::BSH,x) = fill!(parent(b),x)

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

# BiPoSH Yℓ₁ℓ₂st(n₁,n₂)

# Fall back to the generalized SH by default
BiPoSH(args...;kwargs...) = BiPoSH(GSH(),args...;kwargs...)
BiPoSH!(args...;kwargs...) = BiPoSH!(GSH(),args...;kwargs...)

# methods for ordinary and generalized spherical harmonics

function BiPoSH(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s::Integer,t::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	β::Integer,γ::Integer)

	Yℓ₁ℓ₂n₁n₂ = BiPoSH(ASH,ℓ₁,ℓ₂,s,t,(θ₁,ϕ₁),(θ₂,ϕ₂),β:β,γ:γ)
	Yℓ₁ℓ₂n₁n₂[s,t,β,γ]
end

function BiPoSH(::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s::Integer,t::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	β_range::AbstractUnitRange=-1:1,
	γ_range::AbstractUnitRange=-1:1)
	
	Yℓ₁n₁ = Ylmatrix(ℓ₁,(θ₁,ϕ₁),n_range=β_range)
	Yℓ₂n₂ = Ylmatrix(ℓ₂,(θ₂,ϕ₂),n_range=γ_range)
	@assert(δ(ℓ₁,ℓ₂,s),"|ℓ₁-ℓ₂|<=s<=ℓ₁+ℓ₂ not satisfied for ℓ₁=$ℓ₁, ℓ₂=$ℓ₂ and s=$s")
	@assert(abs(t)<=s,"abs(t)<=s not satisfied for t=$t and s=$s")

	Yℓ₁ℓ₂n₁n₂ = BSH{st}(s:s,t:t,β_range,γ_range)

	for γ in γ_range,β in β_range
		for m in -ℓ₁:ℓ₁
			n = t - m
			if abs(n)>ℓ₂
				continue
			end
			Yℓ₁ℓ₂n₁n₂[s,t,β,γ] += clebschgordan(ℓ₁,m,ℓ₂,n,s,t)*Yℓ₁n₁[m,β]*Yℓ₂n₂[n,γ]
		end
	end

	return Yℓ₁ℓ₂n₁n₂
end

BiPoSH(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s::Integer,t::Integer,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH(ASH,ℓ₁,ℓ₂,s,t,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

function BiPoSH(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s_range::AbstractUnitRange,
	args...;t = -maximum(s_range):maximum(s_range),kwargs...)

	SHModes = st(s_range,t)
	BiPoSH(ASH,ℓ₁,ℓ₂,SHModes,args...;kwargs...)
end

function BiPoSH(::GSH,ℓ₁::Integer,ℓ₂::Integer,SHModes::SHModeRange,
	x1::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	x2::Union{Tuple{<:Real,<:Real},<:SphericalPoint},args...;
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	kwargs...)

	β,γ = to_unitrange.((β,γ))

	if (β==0:0) && (γ==0:0)
		# Fall back to the faster ordinary BSH
		BiPoSH(OSH(),ℓ₁,ℓ₂,SHModes,x1,x2,args...;kwargs...)
	end

	Yℓ₁n₁ = zeros(ComplexF64,-ℓ₁:ℓ₁,β)
	Yℓ₂n₂ = zeros(ComplexF64,-ℓ₂:ℓ₂,γ)

	BiPoSH!(GSH(),ℓ₁,ℓ₂,SHModes,x1,x2,Yℓ₁n₁,Yℓ₂n₂,args...;
		β=β,γ=γ,kwargs...,compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true)
end

function BiPoSH(::OSH,ℓ₁::Integer,ℓ₂::Integer,SHModes::SHModeRange,
	x1::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	x2::Union{Tuple{<:Real,<:Real},<:SphericalPoint},args...;
	kwargs...)

	Yℓ₁n₁ = zeros(ComplexF64,-ℓ₁:ℓ₁,0:0)
	Yℓ₂n₂ = zeros(ComplexF64,-ℓ₂:ℓ₂,0:0)

	BiPoSH!(OSH(),ℓ₁,ℓ₂,SHModes,x1,x2,Yℓ₁n₁,Yℓ₂n₂,args...;
		kwargs...,compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true,β=0:0,γ=0:0)
end



function BiPoSH!(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s_range::AbstractUnitRange,
	args...;t=-maximum(s_range):maximum(s_range),kwargs...)

	SHModes = st(s_range,t)
	BiPoSH!(ASH,ℓ₁,ℓ₂,SHModes,args...;kwargs...)
end

function BiPoSH!(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,SHModes::SHModeRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁,Yℓ₂n₂,args...;
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	kwargs...)

	β,γ = to_unitrange.((β,γ))

	BiPoSH_compute!(ASH,ℓ₁,ℓ₂,SHModes,(θ₁,ϕ₁),(θ₂,ϕ₂),
		Yℓ₁n₁,Yℓ₂n₂,β,γ,args...;kwargs...)
end

function BiPoSH!(ASH::AbstractSH,B::BSH,ℓ₁::Integer,ℓ₂::Integer,s_range::AbstractUnitRange,
	args...;t = -maximum(s_range):maximum(s_range),kwargs...)

	SHModes = st(s_range,t)
	BiPoSH!(ASH,B,ℓ₁,ℓ₂,SHModes,args...;kwargs...)
end

function BiPoSH!(ASH::AbstractSH,B::BSH{T},ℓ₁::Integer,ℓ₂::Integer,SHModes::T,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁,Yℓ₂n₂,args...;
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	kwargs...) where {T<:SHModeRange}

	β,γ = to_unitrange.((β,γ))

	BiPoSH_compute!(ASH,B,ℓ₁,ℓ₂,SHModes,(θ₁,ϕ₁),(θ₂,ϕ₂),
		Yℓ₁n₁,Yℓ₂n₂,β,γ,args...;kwargs...)
end

function BiPoSH!(ASH::AbstractSH,B::BSH{T},ℓ₁::Integer,ℓ₂::Integer,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁,Yℓ₂n₂,args...;
	β::Union{Integer,AbstractUnitRange}=-1:1,
	γ::Union{Integer,AbstractUnitRange}=-1:1,
	kwargs...) where {T<:SHModeRange}

	β,γ = to_unitrange.((β,γ))

	BiPoSH_compute!(ASH,B,ℓ₁,ℓ₂,modes(B),(θ₁,ϕ₁),(θ₂,ϕ₂),
		Yℓ₁n₁,Yℓ₂n₂,β,γ,args...;kwargs...)
end

BiPoSH!(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s_range::AbstractUnitRange,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH!(ASH,ℓ₁,ℓ₂,s_range,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

BiPoSH!(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,SHModes::SHModeRange,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH!(ASH,ℓ₁,ℓ₂,SHModes,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

BiPoSH!(ASH::AbstractSH,B::BSH,ℓ₁::Integer,ℓ₂::Integer,s_range::AbstractUnitRange,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) = 
	BiPoSH!(ASH,B,ℓ₁,ℓ₂,s_range,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

BiPoSH!(ASH::AbstractSH,B::BSH{T},ℓ₁::Integer,ℓ₂::Integer,SHModes::T,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) where {T<:SHModeRange} = 
	BiPoSH!(ASH,B,ℓ₁,ℓ₂,SHModes,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

BiPoSH!(ASH::AbstractSH,B::BSH{T},ℓ₁::Integer,ℓ₂::Integer,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...) where {T<:SHModeRange} = 
	BiPoSH!(ASH,B,ℓ₁,ℓ₂,(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)

"""
	BiPoSH!(OSH(),Yℓ′n₁ℓn₂::Matrix{ComplexF64},
	ℓ_range::AbstractUnitRange,SHModes::SHModeRange,args...;kwargs...)

	Compute BiPoSH for a range in ℓ and ℓ′
"""
function BiPoSH!(::OSH,Yℓ′n₁ℓn₂::Matrix{ComplexF64},
	ℓ′ℓ_smax::s′s,SHModes::SHModeRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ′n₁::AbstractMatrix{<:Complex},
	Yℓn₂::AbstractMatrix{<:Complex},
	YSH_n1=nothing,YSH_n2=nothing,
	P=nothing,coeff=nothing;
	CG=nothing,w3j=nothing,
	wig3j_fn_ptr=nothing,
	compute_Yℓ₁n₁=false,compute_Yℓ₂n₂=false)

	lmax = maximum(s_range(ℓ′ℓ_smax))
	l′max = maximum(s′_range(ℓ′ℓ_smax))

	if compute_Yℓ₁n₁ || compute_Yℓ₂n₂
		# Precompute the spherical harmonics
		if isnothing(coeff)
			coeff = SphericalHarmonics.compute_coefficients(l′max)
		end
		if isnothing(P)
			P = SphericalHarmonics.allocate_p(l′max)
		end

		if compute_Yℓ₁n₁
			compute_p!(l′max,cos(θ₁),coeff,P)
			if isnothing(YSH_n1)
				YSH_n1 = SphericalHarmonics.allocate_y(l′max)
			end
			compute_y!(l′max,ϕ₁,P,YSH_n1)
		end

		if compute_Yℓ₂n₂
			compute_p!(lmax,cos(θ₂),coeff,P)
			if isnothing(YSH_n2)
				YSH_n2 = SphericalHarmonics.allocate_y(lmax)
			end
			compute_y!(lmax,ϕ₂,P,YSH_n2)
		end
	end

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	if isnothing(w3j)
		w3j = zeros(lmax+l′max+1)
	end
	if isnothing(CG)
		CG = zeros(0:lmax+l′max)
	end

	@inbounds for (ind,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)

		outputarr = OffsetArray(reshape(
					view(Yℓ′n₁ℓn₂,:,ind),axes(SHModes,1),1,1),
						axes(SHModes,1),0:0,0:0)

		B = BSH(SHModes,outputarr)
		BiPoSH!(OSH(),B,
			ℓ′,ℓ,SHModes,(θ₁,ϕ₁),(θ₂,ϕ₂),
			Yℓ′n₁,Yℓn₂,YSH_n1,YSH_n2;
			CG=CG,w3j=w3j,β=0:0,γ=0:0,
			wig3j_fn_ptr=wig3j_fn_ptr)
	end

	!isnothing(lib) && Libdl.dlclose(lib)

	return Yℓ′n₁ℓn₂
end

function BiPoSH!(::OSH,Yℓ′n₁ℓn₂::Matrix{ComplexF64},
	ℓ_range::AbstractUnitRange,SHModes::SHModeRange,
	args...;kwargs...)

	ℓ′ℓ_smax = s′s(ℓ_range,SHModes)
	BiPoSH!(OSH(),Yℓ′n₁ℓn₂,ℓ′ℓ_smax,SHModes,args...;kwargs...)
end

function BiPoSH!(::OSH,Yℓ′n₁ℓn₂::Matrix{ComplexF64},
	ℓ_range,SHModes::SHModeRange,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...)
	
	BiPoSH!(OSH(),Yℓ′n₁ℓn₂,ℓ_range,SHModes,
		(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)
end

function BiPoSH(::OSH,ℓ′ℓ_smax::s′s,SHModes::SHModeRange,
	x1::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	x2::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	args...;kwargs...)

	Yℓ′n₁ℓn₂ = zeros(ComplexF64,length(SHModes),length(ℓ′ℓ_smax))

	lmax = maximum(s_range(ℓ′ℓ_smax))
	l′max = maximum(s′_range(ℓ′ℓ_smax))

	Yℓ′n₁ = zeros(ComplexF64,-l′max:l′max,0:0)
	Yℓn₂ = zeros(ComplexF64,-lmax:lmax,0:0)

	BiPoSH!(OSH(),Yℓ′n₁ℓn₂,ℓ′ℓ_smax,SHModes,x1,x2,Yℓ′n₁,Yℓn₂,args...;
		kwargs...,compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true)
end

function BiPoSH(::OSH,ℓ_range::AbstractUnitRange,SHModes::SHModeRange,
	args...;kwargs...)

	ℓ′ℓ_smax = s′s(ℓ_range,SHModes)
	BiPoSH(OSH(),ℓ′ℓ_smax,SHModes,args...;kwargs...)
end

"""

	BiPoSH!(OSH(),Yℓ′n₁ℓn₂::Matrix{ComplexF64},Yℓ′n₂ℓn₁::Matrix{ComplexF64},
	ℓ_range::AbstractUnitRange,SHModes::SHModeRange,args...;kwargs...)

	Compute BiPoSH for a range in ℓ and ℓ′ by switching the two points
	Returns Yℓ′n₁ℓn₂ and Yℓ′n₂ℓn₁
"""
function BiPoSH!(::OSH,Yℓ′n₁ℓn₂::Matrix{ComplexF64},Yℓ′n₂ℓn₁::Matrix{ComplexF64},
	ℓ′ℓ_smax::s′s,SHModes::SHModeRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ′n₁::AbstractMatrix{<:Complex},
	Yℓn₁::AbstractMatrix{<:Complex},
	Yℓ′n₂::AbstractMatrix{<:Complex},
	Yℓn₂::AbstractMatrix{<:Complex},
	YSH_n1=nothing,YSH_n2=nothing,
	P=nothing,coeff=nothing;
	CG=nothing,w3j=nothing,
	wig3j_fn_ptr=nothing,
	compute_Yℓ₁n₁=true,
	compute_Yℓ₂n₂=true)

	lmax = maximum(s_range(ℓ′ℓ_smax))
	l′max = maximum(s′_range(ℓ′ℓ_smax))
	l_highest = max(l′max,lmax)

	if compute_Yℓ₁n₁ || compute_Yℓ₂n₂
		# Precompute the spherical harmonics
		if isnothing(coeff)
			coeff = SphericalHarmonics.compute_coefficients(l_highest)
		end
		if isnothing(P)
			P = SphericalHarmonics.allocate_p(l_highest)
		end

		if compute_Yℓ₁n₁
			compute_p!(l_highest,cos(θ₁),coeff,P)
			if isnothing(YSH_n1)
				YSH_n1 = SphericalHarmonics.allocate_y(l_highest)
			end
			compute_y!(l_highest,ϕ₁,P,YSH_n1)
		end

		if compute_Yℓ₂n₂
			compute_p!(l_highest,cos(θ₂),coeff,P)
			if isnothing(YSH_n2)
				YSH_n2 = SphericalHarmonics.allocate_y(l_highest)
			end
			compute_y!(l_highest,ϕ₂,P,YSH_n2)
		end
	end

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	if isnothing(w3j)
		w3j = zeros(lmax+l′max+1)
	end
	if isnothing(CG)
		CG = zeros(0:lmax+l′max)
	end

	@inbounds for (indℓ′ℓ,(ℓ′,ℓ)) in enumerate(ℓ′ℓ_smax)

		# In one pass we can compute Yℓ′n₁ℓn₂ and Yℓn₂ℓ′n₁

		Yℓ′n₁ℓn₂_st = OffsetArray(reshape(
						view(Yℓ′n₁ℓn₂,:,indℓ′ℓ),axes(SHModes,1),1,1),
							axes(SHModes,1),0:0,0:0)

		Yℓ′n₂ℓn₁_st = OffsetArray(reshape(
					view(Yℓ′n₂ℓn₁,:,indℓ′ℓ),axes(SHModes,1),1,1),
						axes(SHModes,1),0:0,0:0)

		# We use Yℓ′n₂ℓn₁ = (-1)^(ℓ+ℓ′+s) Yℓn₁ℓ′n₂
		# and Yℓ′n₁ℓn₂ = (-1)^(ℓ+ℓ′+s) Yℓn₂ℓ′n₁
		# Precomputation of the RHS would have happened if ℓ′<ℓ, 
		# as the modes are sorted in order of increasing ℓ

		if (ℓ,ℓ′) in ℓ′ℓ_smax && ℓ′<ℓ
			# In this case Yℓn₁ℓ′n₂ and Yℓn₂ℓ′n₁ have already been computed
			# This means we can evaluate Yℓ′n₂ℓn₁ and Yℓ′n₁ℓn₂ using the formulae
			# presented above

			indℓℓ′ = modeindex(ℓ′ℓ_smax,(ℓ,ℓ′))

			Yℓn₁ℓ′n₂_st = OffsetArray(reshape(
						view(Yℓ′n₁ℓn₂,:,indℓℓ′),axes(SHModes,1),1,1),
							axes(SHModes,1),0:0,0:0)

			Yℓn₂ℓ′n₁_st = OffsetArray(reshape(
					view(Yℓ′n₂ℓn₁,:,indℓℓ′),axes(SHModes,1),1,1),
						axes(SHModes,1),0:0,0:0)

			for (indst,(s,t)) in enumerate(SHModes)
				Yℓ′n₂ℓn₁_st[indst,0,0] = (-1)^(ℓ+ℓ′+s)*Yℓn₁ℓ′n₂_st[indst,0,0]
				Yℓ′n₁ℓn₂_st[indst,0,0] = (-1)^(ℓ+ℓ′+s)*Yℓn₂ℓ′n₁_st[indst,0,0]
			end

		else
			# Default case, where we need to evaluate both

			BiPoSH!(OSH(),BSH(SHModes,Yℓ′n₂ℓn₁_st),
				ℓ′,ℓ,(θ₂,ϕ₂),(θ₁,ϕ₁),
				Yℓ′n₂,Yℓn₁,YSH_n2,YSH_n1;
				CG=CG,w3j=w3j,β=0:0,γ=0:0,
				wig3j_fn_ptr=wig3j_fn_ptr)

			BiPoSH!(OSH(),BSH(SHModes,Yℓ′n₁ℓn₂_st),
				ℓ′,ℓ,(θ₁,ϕ₁),(θ₂,ϕ₂),
				Yℓ′n₁,Yℓn₂,YSH_n1,YSH_n2;
				CG=CG,w3j=w3j,β=0:0,γ=0:0,
				wig3j_fn_ptr=wig3j_fn_ptr)
		end

	end

	!isnothing(lib) && Libdl.dlclose(lib)

	return Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁
end

function BiPoSH!(::OSH,Yℓ′n₁ℓn₂::Matrix{ComplexF64},Yℓ′n₂ℓn₁::Matrix{ComplexF64},
	ℓ′ℓ_smax::s′s,SHModes::SHModeRange,
	x1::SphericalPoint,x2::SphericalPoint,args...;kwargs...)
	
	BiPoSH!(OSH(),Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,ℓ′ℓ_smax,SHModes,
		(x1.θ,x1.ϕ),(x2.θ,x2.ϕ),args...;kwargs...)
end

function BiPoSH_n1n2_n2n1(ASH::AbstractSH,ℓ_range::AbstractUnitRange,SHModes::SHModeRange,
	x1::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	x2::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	args...;kwargs...)

	ℓ′ℓ_smax = s′s(ℓ_range,SHModes.smax)
	BiPoSH_n1n2_n2n1(ASH,ℓ′ℓ_smax,SHModes,x1,x2,args...;kwargs...)
end

function BiPoSH_n1n2_n2n1(ASH::AbstractSH,ℓ′ℓ_smax::s′s,SHModes::SHModeRange,
	x1::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	x2::Union{Tuple{<:Real,<:Real},<:SphericalPoint},
	args...;kwargs...)

	Yℓ′n₁ℓn₂ = zeros(ComplexF64,length(SHModes),length(ℓ′ℓ_smax))
	Yℓ′n₂ℓn₁ = similar(Yℓ′n₁ℓn₂); fill!(Yℓ′n₂ℓn₁,0)

	lmax = maximum(s_range(ℓ′ℓ_smax))
	l′max = maximum(s′_range(ℓ′ℓ_smax))

	Yℓ′n₁ = zeros(ComplexF64,-l′max:l′max,0:0)
	Yℓn₂ = zeros(ComplexF64,-lmax:lmax,0:0)

	Yℓ′n₂ = similar(Yℓ′n₁); fill!(Yℓ′n₂,0)
	Yℓn₁ = similar(Yℓn₂); fill(Yℓn₁,0)

	BiPoSH!(ASH,Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,ℓ′ℓ_smax,SHModes,
		x1,x2,Yℓ′n₁,Yℓn₁,Yℓ′n₂,Yℓn₂,args...;
		kwargs...,compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true)

	Yℓ′n₁ℓn₂,Yℓ′n₂ℓn₁,ℓ′ℓ_smax
end

# The actual functions that do the calculation for one pair of (ℓ₁,ℓ₂) and 
# (θ₁,ϕ₁) and (θ₂,ϕ₂). The BiPoSH! functions call these.

function BiPoSH_compute!(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,s_range::AbstractUnitRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁::AbstractArray,Yℓ₂n₂::AbstractArray,
	β::AbstractUnitRange=-1:1,
	γ::AbstractUnitRange=-1:1,
	t_range::AbstractUnitRange=-last(s_range):last(s_range),
	args...;kwargs...)

	s_ℓ₁ℓ₂ = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_range = intersect(s_ℓ₁ℓ₂,s_range)
	t_range = intersect(-maximum(s_range):maximum(s_range),t_range)

	Yℓ₁ℓ₂n₁n₂ = BSH{st}(s_range,t_range,β=β,γ=γ)

	BiPoSH_compute!(ASH,Yℓ₁ℓ₂n₁n₂,ℓ₁,ℓ₂,s_range,(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ₁n₁,Yℓ₂n₂,
		β,γ,t_range,args...;kwargs...)
end

function BiPoSH_compute!(ASH::AbstractSH,ℓ₁::Integer,ℓ₂::Integer,SHModes::SHModeRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁::AbstractArray,Yℓ₂n₂::AbstractArray,
	β::AbstractUnitRange=-1:1,
	γ::AbstractUnitRange=-1:1,
	args...;kwargs...)

	B = BSH(SHModes,zeros(ComplexF64,axes(SHModes,1),β,γ))

	BiPoSH_compute!(ASH,B,ℓ₁,ℓ₂,SHModes,(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ₁n₁,Yℓ₂n₂,
		β,γ,args...;kwargs...)
end

function BiPoSH_compute!(ASH::AbstractSH,B::BSH{st},ℓ₁::Integer,ℓ₂::Integer,
	s::AbstractUnitRange,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁::AbstractArray,Yℓ₂n₂::AbstractArray,
	β::AbstractUnitRange=-1:1,
	γ::AbstractUnitRange=-1:1,
	t::AbstractUnitRange=-maximum(s):maximum(s),
	args...;kwargs...)

	BiPoSH_compute!(ASH,B,ℓ₁,ℓ₂,st(s,t),
	(θ₁,ϕ₁),(θ₂,ϕ₂),Yℓ₁n₁,Yℓ₂n₂,β,γ,args...;kwargs...)
end

function BiPoSH_compute!(ASH::AbstractSH,
	Yℓ₁ℓ₂n₁n₂::BSH{st},ℓ₁::Integer,ℓ₂::Integer,
	stmodes::st,
	(θ₁,ϕ₁)::Tuple{<:Real,<:Real},
	(θ₂,ϕ₂)::Tuple{<:Real,<:Real},
	Yℓ₁n₁::AbstractMatrix{<:Complex},
	Yℓ₂n₂::AbstractMatrix{<:Complex},
	β::AbstractUnitRange=-1:1,
	γ::AbstractUnitRange=-1:1,
	dℓ₁::Union{Nothing,AbstractVecOrMat{<:Number}}=nothing,
	dℓ₂::Union{Nothing,AbstractVecOrMat{<:Number}}=nothing;
	CG=nothing,w3j=nothing,
	compute_dℓ₁=true,compute_dℓ₂=true,
	compute_Yℓ₁n₁=true,compute_Yℓ₂n₂=true,
	wig3j_fn_ptr=nothing)

	if compute_Yℓ₁n₁
		Ylmatrix!(ASH,Yℓ₁n₁,dℓ₁,ℓ₁,(θ₁,ϕ₁),n_range=β,
			compute_d_matrix= !isnothing(dℓ₁) && compute_dℓ₁)
	end
	if compute_Yℓ₂n₂
		Ylmatrix!(ASH,Yℓ₂n₂,dℓ₂,ℓ₂,(θ₂,ϕ₂),n_range=γ,
			compute_d_matrix= !isnothing(dℓ₂) && compute_dℓ₂ && (dℓ₂ !== dℓ₁) )
	end

	fill!(Yℓ₁ℓ₂n₁n₂,zero(ComplexF64))

	s_ℓ₁ℓ₂ = abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂
	s_valid = intersect(s_ℓ₁ℓ₂,s_range(stmodes),s_range(Yℓ₁ℓ₂n₁n₂))
	t_valid = intersect(-maximum(s_valid):maximum(s_valid),
		t_range(stmodes),t_range(Yℓ₁ℓ₂n₁n₂))
	β_valid = intersect(β,axes(parent(Yℓ₁ℓ₂n₁n₂),2))
	γ_valid = intersect(γ,axes(parent(Yℓ₁ℓ₂n₁n₂),3))

	lib = nothing

	if isnothing(wig3j_fn_ptr)
		lib=Libdl.dlopen(joinpath(dirname(pathof(WignerD)),"shtools_wrapper.so"))
		wig3j_fn_ptr=Libdl.dlsym(lib,:wigner3j_wrapper)
	end

	if isnothing(w3j)
		w3j = zeros(ℓ₁+ℓ₂+1)
	end
	if isnothing(CG)
		CG = zeros(abs(ℓ₁-ℓ₂):ℓ₁+ℓ₂)
	end

	@inbounds for γ in γ_valid
		
		Yℓ₂n₂γ = @view Yℓ₂n₂[:,γ]

		@inbounds for β in β_valid
			
			Yℓ₁n₁β = @view Yℓ₁n₁[:,β]
			
			Yℓ₁ℓ₂n₁n₂βγ = view(Yℓ₁ℓ₂n₁n₂,:,β,γ)
			
			@inbounds for t in t_valid
				
				Yℓ₁ℓ₂n₁n₂βγt = view(Yℓ₁ℓ₂n₁n₂βγ,:,t)
				
				srange_t = s_valid_range(Yℓ₁ℓ₂n₁n₂,t)
				srange_t_valid = intersect(srange_t,s_valid)
				
				@inbounds for m in -ℓ₁:ℓ₁
		
					n = t - m
					if abs(n) > ℓ₂
						continue
					end

					CG_ℓ₁mℓ₂nst!(ℓ₁,m,ℓ₂,t,CG,w3j;wig3j_fn_ptr=wig3j_fn_ptr)

					Yℓ₁n₁βYℓ₂n₂γ = Yℓ₁n₁β[m]*Yℓ₂n₂γ[n]

					@inbounds for s in srange_t_valid
						s_ind = searchsortedfirst(srange_t,s)
						Yℓ₁ℓ₂n₁n₂βγt[s_ind] += CG[s]*Yℓ₁n₁βYℓ₂n₂γ
					end
				end
			end
		end
	end

	if !isnothing(lib)
		Libdl.dlclose(lib)
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

function CG_ℓ₁mℓ₂nst!(ℓ₁,m,ℓ₂,t,CG;wig3j_fn_ptr=nothing)
	n = t-m
	smin = max(abs(ℓ₁-ℓ₂),abs(t))
	smax = ℓ₁ + ℓ₂
	w3j = Wigner3j(ℓ₁,ℓ₂,m,n;wig3j_fn_ptr=wig3j_fn_ptr)
	@inbounds for (ind,s) in enumerate(smin:smax)
		CG[s] = w3j[ind]*√(2s+1)*(-1)^(ℓ₁-ℓ₂)
	end
	return CG
end

function CG_ℓ₁mℓ₂nst!(ℓ₁,m,ℓ₂,t,CG,w3j;wig3j_fn_ptr=nothing)
	n = t-m
	smin = max(abs(ℓ₁-ℓ₂),abs(t))
	smax = ℓ₁ + ℓ₂
	Wigner3j!(w3j,ℓ₁,ℓ₂,m,n;wig3j_fn_ptr=wig3j_fn_ptr)
	@inbounds for (ind,s) in enumerate(smin:smax)
		CG[s] = w3j[ind]*√(2s+1)*(-1)^(ℓ₁-ℓ₂)
	end
	return CG
end

include("./precompile.jl")

end

