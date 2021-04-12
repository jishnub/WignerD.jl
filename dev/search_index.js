var documenterSearchIndex = {"docs":
[{"location":"","page":"WignerD","title":"WignerD","text":"DocTestSetup  = quote\n    using WignerD\nend","category":"page"},{"location":"#WignerD.jl","page":"WignerD","title":"WignerD.jl","text":"","category":"section"},{"location":"","page":"WignerD","title":"WignerD","text":"Modules = [WignerD]","category":"page"},{"location":"#WignerD.Equator","page":"WignerD","title":"WignerD.Equator","text":"Equator\n\nAlias for θ = π/2 in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#WignerD.NorthPole","page":"WignerD","title":"WignerD.NorthPole","text":"NorthPole\n\nAlias for θ = 0 in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#WignerD.SouthPole","page":"WignerD","title":"WignerD.SouthPole","text":"SouthPole\n\nAlias for θ = π in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#WignerD.wignerD","page":"WignerD","title":"WignerD.wignerD","text":"wignerD(j, α::Real, β::Real, γ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])\n\nEvaluate the Wigner D-matrix with elements D^j_mn(αβγ) for the angular momentum j and the Euler angles α, β and γ. The momentum j may be an integer or a half-integer, and must be non-negative. Optionally the pre-allocated matrix Jy may be provided, which must be a ComplexF64 matrix of size (2j+1, 2j+1), and may be overwritten during the calculation.\n\n\n\n\n\n","category":"function"},{"location":"#WignerD.wignerD!","page":"WignerD","title":"WignerD.wignerD!","text":"wignerD!(D, j, α::Real, β::Real, γ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])\n\nEvaluate the Wigner D-matrix with elements D^j_mn(αβγ) for the angular momentum j and the Euler angles α, β and γ, and store the result in D. The momentum j may be an integer or a half-integer, and must be non-negative. Optionally the pre-allocated matrix Jy may be provided, which must be a ComplexF64 matrix of size (2j+1, 2j+1), and may be overwritten during the calculation.\n\n\n\n\n\n","category":"function"},{"location":"#WignerD.wignerd","page":"WignerD","title":"WignerD.wignerd","text":"wignerd(j, θ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])\n\nEvaluate the Wigner d-matrix with elements d^j_mn(θ) for the angular momentum j and the angle θ. The momentum j may be an integer or a half-integer, and must be non-negative. Optionally the pre-allocated matrix Jy may be provided, which must be a ComplexF64 matrix of size (2j+1, 2j+1), and may be overwritten during the calculation.\n\n\n\n\n\n","category":"function"},{"location":"#WignerD.wignerd!","page":"WignerD","title":"WignerD.wignerd!","text":"wignerd!(d, j, θ::Real, [Jy = zeros(ComplexF64, 2j+1, 2j+1)])\n\nEvaluate the Wigner d-matrix with elements d^j_mn(θ) for the angular momentum j and the angle θ, and store the result in d. The momentum j may be an integer or a half-integer, and must be non-negative. Optionally the pre-allocated matrix Jy may be provided, which must be a ComplexF64 matrix of size (2j+1, 2j+1), and may be overwritten during the calculation.\n\n\n\n\n\n","category":"function"},{"location":"#WignerD.wignerdjmn-Tuple{Any, Any, Any, Real}","page":"WignerD","title":"WignerD.wignerdjmn","text":"wignerdjmn(j, m, n, θ::Real)\n\nEvaluate the Wigner d-matrix element d^j_mn(θ).\n\n\n\n\n\n","category":"method"}]
}
