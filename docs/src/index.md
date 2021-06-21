```@meta
DocTestSetup  = quote
    using WignerD
end
```

# WignerD.jl

## Definition

The Wigner D matrix is the matrix representation of the rotation operator in the eigenbasis of the angular momentum operator ``J_z``. Representing a general eigenvector of ``J_z`` as ``\left|jm\right\rangle`` and given a rotation operator ``U(R)``, we obtain ``D_{mm^{\prime}}^{j}\left(R\right)=\left\langle jm\right|U\left(R\right)\left|jm^{\prime}\right\rangle``. If we represent the rotation in terms of Euler angles, we obtain ``U\left(R\right)=e^{-i J_z \alpha}e^{-i J_y \beta}e^{-i J_z \gamma}``. In this representation, we obtain

```math
\begin{aligned}
D_{mm^{\prime}}^{j}\left(\alpha,\beta,\gamma\right)&=e^{-i\left(m\alpha+m^{\prime}\gamma\right)}\left\langle jm\right|e^{-iJ_{y}\beta}\left|jm^{\prime}\right\rangle \\&=e^{-i\left(m\alpha+m^{\prime}\gamma\right)}d_{mm^{\prime}}^{j}\left(\beta\right)
\end{aligned}
```

where ``d_{mm^{\prime}}^{j}\left(\beta\right)`` is the Wigner d matrix.

We use the phase convention of Varshalovich et al. (1988). In this conevntion, wavefunctions transform under a rotation of coordinate frames as

```math
\psi_{jm^{\prime}}\left(\theta^{\prime},\phi^{\prime},\sigma^{\prime}\right)=\sum_{m}\psi_{jm}\left(\theta,\phi,\sigma\right)D_{mm^{\prime}}^{j}\left(\alpha,\beta,\gamma\right).
```
where ``\alpha``, ``\beta`` and ``\gamma`` are the Euler angles corresponding to the rotation, ``(\theta,\phi)`` and ``(\theta^\prime,\phi^\prime)`` are the polar coordinates in the initial and rotated coordinate systems ``S`` and ``S^{\prime}``, and ``\sigma`` and ``\sigma^{\prime}`` are the spin variables in ``S`` and ``S^{\prime}`` respectively.

In this package we evaluate the Wigner d matrix following [Feng (2015)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.043307). We diagonalize the operator ``J_y`` to obtain its eigenbasis ``\left|j _{y}n\right\rangle``. In terms of these, we may obtain

```math
\begin{aligned}
d_{mm^{\prime}}^{\ell}\left(\beta\right)&=\left\langle jm\right|e^{-iJ_{y}\beta}\left|jm^{\prime}\right\rangle \\&=\left\langle jm|j_{y}p\right\rangle \left\langle j_{y}p\right|e^{-iJ_{y}\beta}\left|j_{y}n\right\rangle \left\langle j_{y}n|jm^{\prime}\right\rangle \\&=e^{-in\beta}\left\langle jm|j_{y}n\right\rangle \left\langle j_{y}n|jm^{\prime}\right\rangle .
\end{aligned}
```

## Rotations of unit vectors

The rotation of a vector may be described in terms of a rotation operator ``U(R)`` as ``U(R)\,\mathbf{x}=\mathbf{y}``. In particular, for the Cartesian unit vectors ``(\mathbf{e}_x,\,\mathbf{e}_y,\mathbf{e}_z)`` we obtain ``U(R)\,\mathbf{e}_i = \mathbf{e}^{\prime}_i``, where ``\mathbf{e}^{\prime}_i`` are the set of rotated unit vectors. In index notation, we may express this as a matrix relation ``R_{ij}\,\mathbf{e}_i = \mathbf{e}^{\prime}_j``. Note that the rows of ``R`` are summed over here, and not columns. The matrix elements ``R_{ij}=\left\langle \mathbf{e}_{i}\right|U\left(R\right)\left|\mathbf{e}_{j}\right\rangle`` are given by ``R_{ij}=\left\langle \mathbf{e}_{i}|\mathbf{e}_{j}^{\prime}\right\rangle``, and the operator ``U(R)`` may be expressed in the Cartesian basis as ``U\left(R\right)=\left|\mathbf{e}_{j}^{\prime}\right\rangle \left\langle \mathbf{e}_{j}\right|=\left\langle \mathbf{e}_{i}|\mathbf{e}_{j}^{\prime}\right\rangle \left|\mathbf{e}_{i}\right\rangle \left\langle \mathbf{e}_{j}\right|``.

Often in the analysis of rotation and spherical harmonics, it is convenient to switch to a basis that is referred to as the "spherical covariant" basis by Varshalovich, and is related to the Cartesian basis through a unitary transformation:

```math
\begin{aligned}
\chi_{-1} & =\frac{1}{\sqrt{2}}\left(\mathbf{e}_{x}-i\mathbf{e}_{y}\right),\\
\chi_{0} & =\mathbf{e}_{z},\\
\chi_{1} & =-\frac{1}{\sqrt{2}}\left(\mathbf{e}_{x}+i\mathbf{e}_{y}\right).
\end{aligned}
```

These vectors form an eigenbasis of the angular momentum operator ``J_z`` for ``j=1``. The rotation of these are also carried out by the same operator: ``U(R)\,\chi_ν = \chi^{\prime}_ν``. In index notation this may be represented as ``D^{1}(R)_{\nu\mu}\,\chi_ν = \chi^{\prime}_\mu``, where, as before, the rows of ``D`` are summed over. The Wigner D matrix ``D^{1}(R)`` is therefore the matrix representation of the rotation operator in the spherical basis ``\chi_ν``.

If we denote the transformation between the Cartesian and spherical covariant bases as a matrix relation ``\chi_μ = U_{\mu i} \mathbf{e}_i``, the relation between the Wigner D matrix ``D^{1}_{\mu\nu}(R)`` and the Cartesian Rotation matrix ``R_{ij}`` is

```math
(D^{1}(R))^* = U R U^{\dagger}
```

The conjugation here is a consequence of the fact that ``U`` has its columns summed over, whereas ``R`` and ``D`` have their rows summed over. If we choose the transpose ``V=U^T``, we may rewrite the relation as

```math
D^{1}(R) = V^T R V
```

We may also derive this using the bra-ket notation as:

```math
\begin{aligned}
D_{\mu\nu}^{1}\left(R\right) & =\left\langle \chi_{\mu}\right|U\left(R\right)\left|\chi_{\nu}\right\rangle \\
 & =\left\langle \chi_{\mu}|\mathbf{e}_{i}\right\rangle \left\langle \mathbf{e}_{i}\right|U\left(R\right)\left|\mathbf{e}_{j}\right\rangle \left\langle \mathbf{e}_{j}|\chi_{\nu}\right\rangle \\
 & =U_{i\mu}^{\dagger}R_{ij}U_{\nu j}\\
 & =U_{\mu i}^{*}R_{ij}U_{j\nu}^{T}\\
 & =\left[U^{*}RU^{T}\right]_{\mu\nu}
\end{aligned}
```

## Rotation of Spherical harmonics

Spherical harmonics ``Y_{ℓm}(\hat{n})`` form an eigenbasis of ``J_z`` for ``j=\ell``. We assume that a rotation ``R`` takes the coordinate frame ``S`` to ``S^\prime``. We assume that a point has coordinates ``\hat{n}=(\theta,\phi)`` in ``S`` and ``\hat{n}^\prime=(\theta^\prime,\phi^\prime)`` in ``S^\prime``, where ``\hat{n}^\prime=R^{-1}\hat{n}``. Under this rotation, spherical harmonics transform as

```math
\begin{aligned}
Y_{\ell m^{\prime}}\left(\hat{n}^{\prime}\right)&=\sum_{m}D_{mm^{\prime}}^{\ell}\left(R\right)Y_{\ell m}\left(\hat{n}\right)\\&=\sum_{m}D_{mm^{\prime}}^{\ell}\left(R\right)Y_{\ell m}\left(R^{-1}\hat{n}^{\prime}\right).
\end{aligned}
```

## Reference

```@autodocs
Modules = [WignerD]
```
