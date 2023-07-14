using CUDA
#using CUDA: i32

# CUDA kernel. When calling, use @cuda.
function AdvectionFV2DCudaKernel!(fF,cF,uF,vF,dXdxI,J,w,uC,vC)
  Nx = size(cF,3)
  Ny = size(cF,4)
  n = size(cF,1)

  @. fF = 0.0

  #for ix = 1 : Nx
  ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
  for iy = 1 : Ny
    for i = 1 : n, j = 1 : n
      uC[i,j] = dXdxI[i,j,1,1,ix,iy] *
                uF[i,j,ix,iy] + dXdxI[i,j,1,2,ix,iy] * vF[i,j,ix,iy]
      vC[i,j] = dXdxI[i,j,2,1,ix,iy] *
                uF[i,j,ix,iy] + dXdxI[i,j,2,2,ix,iy] * vF[i,j,ix,iy]
    end
    for j = 1 : n
      for i = 1 : n - 1
        uCL = 0.5 * (uC[i,j] + uC[i+1,j])
        if uCL > 0.0
          fluxLoc = uCL * cF[i,j,ix,iy] * w[j]
        else
          fluxLoc = uCL * cF[i+1,j,ix,iy] * w[j]
        end
        fF[i,j,ix,iy] = fF[i,j,ix,iy] - fluxLoc
        fF[i+1,j,ix,iy] = fF[i+1,j,ix,iy] + fluxLoc
      end
    end
    for i = 1 : n 
      for j = 1 : n - 1
        vCL = 0.5 * (vC[i,j] + vC[i,j+1])
        if vCL > 0.0
          fluxLoc = vCL * cF[i,j,ix,iy] * w[i]
        else
          fluxLoc = vCL * cF[i,j+1,ix,iy] * w[i]
        end
        fF[i,j,ix,iy] = fF[i,j,ix,iy] - fluxLoc
        fF[i,j+1,ix,iy] = fF[i,j+1,ix,iy] + fluxLoc
      end
    end
  end

  nothing

end
