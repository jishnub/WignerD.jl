module WignerD

using OffsetArrays
using LinearAlgebra
using HalfIntegers

export wignerd!
export wignerd
export wignerD!
export wignerD

abstract type SpecialCoLatitudes <: Real end
Base.promote_rule(::Type{<:SpecialCoLatitudes}, T::Type{<:Real}) = promote_type(Float64, T)
Base.AbstractFloat(p::SpecialCoLatitudes) = Float64(p)

"""
    Equator

Alias for `θ = π/2` in spherical polar coordinates.
"""
struct Equator <: SpecialCoLatitudes end
"""
    NorthPole

Alias for `θ = 0` in spherical polar coordinates.
"""
struct NorthPole <: SpecialCoLatitudes end
"""
    SouthPole

Alias for `θ = π` in spherical polar coordinates.
"""
struct SouthPole <: SpecialCoLatitudes end

Base.Float64(::NorthPole) = zero(Float64)
Base.Float64(::SouthPole) = Float64(pi)
Base.Float64(::Equator) = Float64(π/2)

Base.cos(::NorthPole) = one(Float64)
Base.cos(::SouthPole) = -one(Float64)
Base.sin(::Union{NorthPole, SouthPole}) = zero(Float64)

Base.cos(::Equator) = zero(Float64)
Base.sin(::Equator) = one(Float64)

# Returns exp(i α π/2) = cos(α π/2) + i*sin(α π/2) for integer α
function _cis(α::Integer, ::Equator)
    αmod4 = mod(Int(α), 4)
    if αmod4 == 0
        res = ComplexF64(1, 0)
    elseif αmod4 == 1
        res = ComplexF64(0, 1)
    elseif αmod4 == 2
        res = ComplexF64(-1, 0)
    elseif αmod4 == 3
        res = ComplexF64(0, -1)
    end
    return res
end
function _cis(α::HalfInteger, ::Equator)
    αmod4 = mod(Int(2α), 8)
    if αmod4 == 0
        res = ComplexF64(1, 0)
    elseif αmod4 == 1
        res = ComplexF64(1, 1)/√2
    elseif αmod4 == 2
        res = ComplexF64(0, 1)
    elseif αmod4 == 3
        res = ComplexF64(-1, 1)/√2
    elseif αmod4 == 4
        res = ComplexF64(-1, 0)
    elseif αmod4 == 5
        res = ComplexF64(-1, -1)/√2
    elseif αmod4 == 6
        res = ComplexF64(0, -1)
    elseif αmod4 == 7
        res = ComplexF64(1, -1)/√2
    end
    return res
end

#########################################################################
# Wrapper array to translate the indices
#########################################################################

# This is deliberately not exposed, as indexing into half-integer arrays is not supported in general
struct WignerMatrix{T, J <: Union{Integer, HalfInteger}, A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    j :: J
    D :: A

    function WignerMatrix{T, J, A}(j::J, D::A) where {T, J<:Union{Integer, HalfInteger}, A<:AbstractMatrix{T}}
        @assert j >= 0 "j must be >= 0"
        @assert size(D, 1) == size(D, 2) == 2j+1 "size of matrix must be $(2j+1)×$(2j+1)"
        Base.require_one_based_indexing(D)
        new{T, J, A}(j, D)
    end
end

function WignerMatrix(j::Integer, D::AbstractMatrix{T}) where {T}
    D_noof = OffsetArrays.no_offset_view(D)
    WignerMatrix{T, typeof(j), typeof(D_noof)}(j, D_noof)
end
function WignerMatrix(j::Real, D::AbstractMatrix{T}) where {T}
    j_hi = HalfInteger(j)
    D_noof = OffsetArrays.no_offset_view(D)
    WignerMatrix{T, typeof(j_hi), typeof(D_noof)}(j_hi, D_noof)
end

Base.IndexStyle(::Type{<:WignerMatrix{<:Any, <:Any, A}}) where {A} = IndexStyle(A)
Base.parent(w::WignerMatrix) = w.D
Base.size(w::WignerMatrix) = size(parent(w))
Base.axes(w::WignerMatrix) = (OffsetArrays.IdentityUnitRange(-w.j:w.j), OffsetArrays.IdentityUnitRange(-w.j:w.j))
_index(j, m) = (x = Integer(j + m); x + oneunit(x))
# CartesianIndexing
Base.@propagate_inbounds Base.getindex(w::WignerMatrix, m::Real, n::Real) = parent(w)[_index(w.j, m), _index(w.j, n)]
# CartesianIndexing
Base.@propagate_inbounds Base.setindex!(w::WignerMatrix, val, m::Real, n::Real) = (parent(w)[_index(w.j, m), _index(w.j, n)] = val; w)

_unitrange(a::Integer, b::Integer) = a:b
_unitrange(a::Real, b::Real) = HalfInteger(a):HalfInteger(b)
_half_or_one_to(j::Integer) = _unitrange(1, j)
function _half_or_one_to(j::Real)
    r = reverse(2HalfInt(j):-2:1)
    half(first(r)):half(last(r))
end
_indrange(j::Real) = _unitrange(-j, j)
_matrixsize(j::Real) = (Int(2j + 1), Int(2j + 1))

#########################################################################
# Dictionary to cache the eigenvectors of Jy
# stores (2j+1) => eigvecs(Jy(j))
#########################################################################

const JyEigenDict = Dict{UInt, Matrix{ComplexF64}}()

##########################################################################
# Wigner d matrix
##########################################################################

# Matrix elements of Jy in the eigenbasis of Jz
X(j, n) = sqrt((j + n)*(j-n+1))

function coeffi(j, Jy = zeros(ComplexF64, _matrixsize(j)))
    @assert eltype(Jy) == ComplexF64 "preallocated Jy matrix must have an element type of ComplexF64"
    @assert size(Jy) == (2j + 1, 2j + 1) "preallocated Jy matrix must be of size ($(2j+1), $(2j+1))"

    fill!(Jy, zero(eltype(Jy)))
    for (m, i) in enumerate(diagind(Jy, 1))
        Jy[i] = -X(j, -j + m)/2im
    end
    return Hermitian(Jy)
end

_offsetmatrix(j::Integer, D::AbstractMatrix) = OffsetArray(D, -j:j, -j:j)
_offsetmatrix(j::Real, D::AbstractMatrix) = WignerMatrix(j, D)

function Jy_eigen(j, Jy)
    key = UInt(2j + 1)
    if key in keys(JyEigenDict)
        return _indrange(j), _offsetmatrix(j, JyEigenDict[key])
    end

    Jy_filled = coeffi(j, Jy)
    _, v = eigen!(Jy_filled, sortby = identity)

    # Store the eigenvectors along rows
    vp = permutedims(v)
    JyEigenDict[key] = vp
    w = _offsetmatrix(j, vp)
    return _indrange(j), w
end
function Jy_eigen(j)
    key = UInt(2j + 1)
    if key in keys(JyEigenDict)
        return _indrange(j), _offsetmatrix(j, JyEigenDict[key])
    end

    Jy = zeros(ComplexF64, _matrixsize(j))
    Jy_eigen(j, Jy)
end

"""
    wignerdjmn(j, m, n, θ::Real)

Evaluate the Wigner d-matrix element ``d^j_{m,n}(θ)``.
"""
wignerdjmn(j, m, n, θ::Real) = wignerdjmn(j, m, n, θ, Jy_eigen(j)...)
Base.@propagate_inbounds function wignerdjmn(j, m, n, θ::Real, λ, v)
    dj_m_n = zero(ComplexF64)

    for (μ, λμ) in zip(UnitRange(axes(v, 1)), λ)
        dj_m_n += cis(-λμ * θ) * v[μ, m] * conj(v[μ, n])
    end

    real(dj_m_n)
end

wignerdjmn(j, m, n, θ::NorthPole) = wignerdjmn(j, m, n, θ, nothing, nothing)
function wignerdjmn(j, m, n, θ::NorthPole, λ, v)
    (m == n) ? one(Float64) : zero(Float64)
end

wignerdjmn(j, m, n, θ::SouthPole) = wignerdjmn(j, m, n, θ, nothing, nothing)
function wignerdjmn(j, m, n, θ::SouthPole, λ, v)
    (m == -n) ? iseven(Int(j - n)) ? Float64(1) : Float64(-1) : zero(Float64)
end

Base.@propagate_inbounds function wignerdjmn(j, m, n, θ::Equator, λ, v)
    dj_m_n = zero(ComplexF64)

    if !((isodd(Int(j + m)) && n == 0) || (isodd(Int(j + n)) && m == 0))
        for (μ, λμ) in zip(axes(v, 1), λ)
            dj_m_n += _cis(-λμ, θ) * v[μ, m] * conj(v[μ, n])
        end
    end
    real(dj_m_n)
end

Base.@propagate_inbounds function wignerd!(dj, j, θ::NorthPole, λ, v)
    dj_noof = OffsetArrays.no_offset_view(dj)
    for ind in diagind(dj_noof)
        dj_noof[ind] = one(eltype(dj))
    end
    return dj
end

Base.@propagate_inbounds function wignerd!(dj, j, θ::SouthPole, λ, v)
    dj_w = WignerMatrix(j, OffsetArrays.no_offset_view(dj))

    even = true
    for m in _indrange(j)
        dj_w[m, -m] = even ? one(eltype(dj)) : -one(eltype(dj))
        even = !even
    end
    return dj
end

Base.@propagate_inbounds function wignerd!(dj, j, θ, λ, v)
    @boundscheck @assert size(dj) == _matrixsize(j) "size of dj must be $(Int(2j+1))×$(Int(2j+1))"

    dj_w = _offsetmatrix(j, dj)

    for n in _unitrange(-j, 0), m in _unitrange(n, -n)
        dj_w[m, n] = real(wignerdjmn(j, m, n, θ, λ, v))
    end

    # Use symmetries to fill up other terms
    # dj[m,n] = (-1)^(m-n) * dj[-m,-n]
    for n in _half_or_one_to(j)
        for m in _unitrange(-n, n)
            dj_w[m, n] = (-1)^(m-n) * dj_w[-m, -n]
        end
    end

    # dj[m,n] = (-1)^(m-n) * dj[n,m]
    for m in _half_or_one_to(j)
        for n in _unitrange(-m, m)
            dj_w[m, n] = (-1)^(m-n) * dj_w[n, m]
        end
    end

    for m in _unitrange(-j, -1), n in _unitrange(m, -m)
        dj_w[m, n] = dj_w[-n, -m]
    end

    return dj
end

"""
    wignerd!(d, j, θ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])

Evaluate the Wigner d-matrix with elements ``d^j_{m,n}(θ)`` for the angular momentum ``j`` and the angle ``θ``,
and store the result in `d`.
The momentum ``j`` may be an integer or a half-integer, and must be non-negative.
Optionally the pre-allocated matrix `Jy` may be provided,
which must be a `ComplexF64` matrix of size `(2j+1, 2j+1)`, and may be overwritten during the calculation.
"""
Base.@propagate_inbounds function wignerd!(d, j, θ::Real, Jy = zeros(ComplexF64, _matrixsize(j)))
    λ, v = Jy_eigen(j, Jy)
    wignerd!(d, j, θ, λ, v)
    return d
end

"""
    wignerd(j, θ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])

Evaluate the Wigner d-matrix with elements ``d^j_{m,n}(θ)`` for the angular momentum ``j`` and the angle ``θ``.
The momentum ``j`` may be an
integer or a half-integer, and must be non-negative. Optionally the pre-allocated matrix `Jy` may be provided,
which must be a `ComplexF64` matrix of size `(2j+1, 2j+1)`, and may be overwritten during the calculation.
"""
function wignerd(j, θ::Real, Jy = zeros(ComplexF64, _matrixsize(j)))
    d = zeros(_matrixsize(j))
    @inbounds wignerd!(d, j, θ, Jy)
    return d
end

"""
    wignerD!(D, j, α::Real, β::Real, γ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])

Evaluate the Wigner D-matrix with elements ``D^j_{m,n}(α,β,γ)`` for the angular momentum ``j`` and the
Euler angles ``α``, ``β`` and ``γ``, and store the result in `D`.
The momentum ``j`` may be an integer or a half-integer, and must be non-negative.
Optionally the pre-allocated matrix `Jy` may be provided,
which must be a `ComplexF64` matrix of size `(2j+1, 2j+1)`, and may be overwritten during the calculation.
"""
Base.@propagate_inbounds function wignerD!(D, j, α::Real, β::Real, γ::Real, Jy = zeros(ComplexF64, _matrixsize(j)))
    wignerd!(D, j, β, Jy)
    D_w = _offsetmatrix(j, D)
    for n in _indrange(j), m in _indrange(j)
        D_w[m, n] *= cis(-(m * α + n * γ))
    end
    return D
end

"""
    wignerD(j, α::Real, β::Real, γ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])

Evaluate the Wigner D-matrix with elements ``D^j_{m,n}(α,β,γ)`` for the angular momentum ``j`` and the
Euler angles ``α``, ``β`` and ``γ``.
The momentum ``j`` may be an integer or a half-integer, and must be non-negative.
Optionally the pre-allocated matrix `Jy` may be provided,
which must be a `ComplexF64` matrix of size `(2j+1, 2j+1)`, and may be overwritten during the calculation.
"""
function wignerD(j, α::Real, β::Real, γ::Real, Jy = zeros(ComplexF64, _matrixsize(j)))
    D = zeros(ComplexF64, _matrixsize(j))
    @inbounds wignerD!(D, j, α, β, γ, Jy)
    return D
end

end

