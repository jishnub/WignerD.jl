```@meta
DocTestSetup  = quote
    using WignerD
end
```

# WignerD.jl

Wigner D matrices computed using the phase convention of Varshalovich et al. (1988). In this conevntion, wavefucntions transform under rotation as

```math
\psi_{jm}(\hat{n}) = \sum_{m^\prime=-j}^j D^{j*}_{m,m^\prime}(\alpha,\beta,\gamma) \psi_{jm^\prime}(\hat{n}^\prime),
```
where ``\alpha``, ``\beta`` and ``\gamma`` are the Euler angles corresponding to the rotation. The Wigner D -matrix is related to the d-matrix through

```math
D^{j}_{m,m^\prime}(\alpha,\beta,\gamma) = d^j_{m,m^\prime}(\beta) \exp(-i(m\alpha + m^\prime\gamma))
```

```@autodocs
Modules = [WignerD]
```
