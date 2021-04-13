# WignerD.jl

[![CI](https://github.com/jishnub/WignerD.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/jishnub/WignerD.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jishnub/WignerD.jl/branch/master/graph/badge.svg?token=CSmEtdY3o6)](https://codecov.io/gh/jishnub/WignerD.jl)
[![docs:stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/WignerD.jl/stable)
[![docs:dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/WignerD.jl/dev)

Wigner d and D matrices using the exact diagonalization algorithm of Feng (2015), following the phase convention of Varshalovich et al. (1988).

# Usage

```julia
julia> wignerd(0.5, 0)
2×2 Matrix{Float64}:
 1.0  -0.0
 0.0   1.0

julia> wignerd(1, pi/3)
3×3 Matrix{Float64}:
  0.75       0.612372  0.25
 -0.612372   0.5       0.612372
  0.25      -0.612372  0.75

julia> WignerD.wignerdjmn(1, 1, 1, pi/3)
0.7500000000000004

julia> wignerD(1, 0, pi/3, pi/2)
3×3 Matrix{ComplexF64}:
 4.59243e-17+0.75im       0.612372-0.0im  1.53081e-17-0.25im
 -3.7497e-17-0.612372im        0.5+0.0im   3.7497e-17-0.612372im
 1.53081e-17+0.25im      -0.612372+0.0im  4.59243e-17-0.75im

julia> WignerD.wignerDjmn(1, 1, 1, 0, pi/3, pi/2)
4.592425496802577e-17 - 0.7500000000000004im
```
