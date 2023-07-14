# Advection2D.jl
Solves the two-dimensional advection equation in a Cartesian domain.

$$
\frac{\partial c}{\partial t} + \frac{\partial uc}{\partial x} + \frac{\partial vc}{\partial y} =0
$$

where $(u,v)$ is a given velocity field.

## Usage
You need Julia -- consider
[juliaup](https://github.com/JuliaLang/juliaup#mac-and-linux) for installing.

Install dependencies by running `julia` in the repository root and entering the
following code in the REPL:

```julia
import Pkg; Pkg.add("CairoMakie")
```

It'll take a few minutes to download and precompile packages.

For CUDA:

```julia
import Pkg; Pkg.add("CUDA")
```

## Background
This code is intended to develop positivity preserving advection schemes.
Therefore the data structure resembles what we have normally in a spectral element code. The simulation domain is divided in a number of macro conforming quadrilateral elements. Inside each element you have a number of degrees of freedom. For two-dimensional quadrilaterals a tensor product ansatz of one dimensional Lagrange function is used defined on the
interval $[-1,1]$.

$$ -1 \leq \xi_{0}  < \xi_{1}  < \ldots < \xi_N \leq  1$$

$$
\phi_{k}(\xi) = \prod_{l=0,l \neq k}^{l=N}\frac{\xi-\xi_{l}}{\xi_{k}-\xi_{l}}
$$

Tri numerical methods are implemented on the same number of degree of freedoms, a pure finite volume method, a pure continuous spectral element method and combined method. Since the first two methods can be written in the same flux form they can be combined by a convex combination of both fluxes. The be+lending factor is computed locally for each flux and is determined from the smoothness of the tracer profile,

To run the code the user can choose the domain size (Lx,Ly), the number of grid boxes (Nx,Ny) in the x and y direction and the order n of the element function.
