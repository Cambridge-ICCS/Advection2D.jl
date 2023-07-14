using LinearAlgebra
using CairoMakie
include("GaussLobattoQuad.jl")
include("Lagrange.jl")
include("DLagrange.jl")
include("Jacobi2D.jl")
include("Initial2D.jl")
include("AdvectionFV2D.jl")
include("Visualization.jl")
include("DSS.jl")
include("Flux.jl")

function TestAdvectionFV2D()
  
  n = 4
  Nx = 40
  Ny = 40
  Lx = 1000
  Ly = 1000
  Pert = 0.0
  Case = "Cube"
  
  wF,xF = GaussLobattoQuad(n)
  
  xe = zeros(n+1)
  xe[1] = -1
  for i = 2 : n
    xe[i] = xe[i-1] + 2 / n
  end
  xe[n+1] = 1
  
  xec = zeros(n)
  xec[1] = -1+1/n
  for i = 2 : n-1
    xec[i] = xec[i-1] + 2 / n
  end
  xec[n] = 1-1/n
  
  DF = zeros(n+1,n+1)
  for i = 1 : n + 1
    for j = 1 : n + 1
      DF[i,j] = DLagrange(xF[i],xF,j)
    end
  end
  
  DW = -inv(diagm(wF)) * DF' * diagm(wF)
  
  
  IntF2EC = zeros(n,n+1)
  for j = 1 : n + 1
    for i = 1 : n
      IntF2EC[i,j] = Lagrange(xec[i],xF,j)
    end
  end
  
  Q = diagm(wF) * DF
  S = Q - Q'
  
  P = zeros(Nx+1,Ny+1,2)
  dx = Lx / Nx
  dy = Ly / Ny
  @views @. P[1,:,1] = 0
  @views @. P[:,1,2] = 0
  for ix = 1 : Nx
    @views @. P[ix+1,:,1] = P[ix,:,1] + dx
  end
  for iy = 1 : Ny
    @views @. P[:,iy+1,2] = P[:,iy,2] + dy
  end
  for ix = 2 : Nx
    for iy = 2 : Ny
      P[ix,iy,1] = P[ix,iy,1] + Pert * (2.0 * rand() - 1.0) * dx
      P[ix,iy,2] = P[ix,iy,2] + Pert * (2.0 * rand() - 1.0) * dy
    end
  end
  
  
  X = zeros(n+1,n+1,2,Nx,Ny)
  dXdxI = zeros(n+1,n+1,2,2,Nx,Ny)
  J = zeros(n+1,n+1,Nx,Ny)
  for ix = 1 : Nx
    for iy = 1 : Ny
      X[:,:,:,ix,iy],dXdxI[:,:,:,:,ix,iy],J[:,:,ix,iy] = Jacobi2D(P[ix,iy,:],P[ix+1,iy,:],
        P[ix+1,iy+1,:],P[ix,iy+1,:],xF,DF)
    end
  end
  
  cF0 = zeros(n+1,n+1,Nx,Ny)
  uF = ones(n+1,n+1,Nx,Ny)
  vF = ones(n+1,n+1,Nx,Ny)
  for ix = 1 : Nx
    for iy = 1 : Ny
      for j = 1 : n + 1
        for i = 1 : n +1
          cF0[i,j,ix,iy] = Initial2D(X[i,j,1,ix,iy],X[i,j,2,ix,iy],Case)
        end
      end
    end
  end
  Plot2DC(cF0,IntF2EC,"ScalarFV")
  
  
  nIter = 2000
  dtau = 0.005
  cFn = zeros(n+1,n+1,Nx,Ny)
  cFFV = zeros(n+1,n+1,Nx,Ny)
  fF = zeros(n+1,n+1,Nx,Ny)
  @. cFFV = cF0
  @time for iTer = 1 : nIter
    @. cFn = cFFV
    AdvectionFV2D!(fF,cFFV,uF,vF,dXdxI,J,wF)
    @. cFFV = cFn + 1/3 * dtau * fF
    AdvectionFV2D!(fF,cFFV,uF,vF,dXdxI,J,wF)
    @. cFFV = cFn + 1/2 * dtau * fF
    AdvectionFV2D!(fF,cFFV,uF,vF,dXdxI,J,wF)
    @. cFFV = cFn + dtau * fF
  end
  Plot2DC(cFFV,IntF2EC,"ScalarEndeFV")
end
  
  
  
  
  
  
  
  
  
  
  
  
  
