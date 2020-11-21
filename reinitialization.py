# from   dolfin               import *
# from   mpl_toolkits.mplot3d import Axes3D
# from   matplotlib           import cm
# from   matplotlib.ticker    import LinearLocator, FormatStrFormatter

import pdb
import numpy                as np 
# import matplotlib.pyplot    as plt 
# import matplotlib.image     as mpimg
import time                 as tm


def reinit(nz, nx, phi_small):
     
     # activeplot = 3
     
    #  xx  = np.zeros((1,nx+1), dtype='float32')
    #  yy  = np.zeros((nz+1,1), dtype='float32')
    #  yy0 = np.ones((nz+1,1), dtype='float32')
    #  xx0 = np.ones((1,nx+1), dtype='float32')
     
    #  for m in range(0,nx+1):
    #      xx[0,m] = np.double(m)/nx
    #  for m in range(0,nz+1):
    #      yy[m,0] = np.double(m)/nz
     
    #  XX = np.kron(xx,yy0)
    #  YY = np.kron(yy,xx0)
     
     N   = np.max((nz,nx))+2
     h   = 1./N
     eps = 1e-4
     #############################################
     # Here we extend the level set function
     # because we take the gradient afterward so 
     # so we need to define the level set one a larger grid
     
     phi          = np.zeros((nz+1,nx+1), dtype='float32')
     phi[1:nz,1:nx] = phi_small[0:nz-1,0:nx-1]
     
     phi[0,0:nx+1] = 2*phi[1,0:nx+1]  -phi[2,0:nx+1]
     phi[nz,0:nx+1] = 2*phi[nz-1,0:nx+1]-phi[nz-2,0:nx+1]
     
     phi[0:nz+1,0] = 2*phi[0:nz+1,1]  -phi[0:nz+1,2]
     phi[0:nz+1,nx] = 2*phi[0:nz+1,nx-1]-phi[0:nz+1,nx-2]
      
     
     Dxp = np.zeros((nz,nx), dtype='float32')
     Dyp = np.zeros((nz,nx), dtype='float32')
     
     Dxm = np.zeros((nz,nx), dtype='float32')
     Dym = np.zeros((nz,nx), dtype='float32')
     
     Dxs = np.zeros((nz,nx), dtype='float32')
     Dys = np.zeros((nz,nx), dtype='float32')
     
     ##########################
     # Be careful here, 
     # the x-axis is in columns
     # the y-axis is in rows
     
     for m in range(0,nx-1):
         Dxp[0:nz-1,m]   = phi[1:nz,m+1]*nz - phi[1:nz,m]*nz
     for m in range(0,nz-1):
         Dyp[m,0:nx-1]   = phi[m+1,1:nx]*nz - phi[m,1:nx]*nz
     for m in range(1,nx):
         Dxm[0:nz-1,m-1] = phi[1:nz,m]*nx   - phi[1:nz,m-1]*nx
         Dxs[0:nz-1,m-1] = phi[1:nz,m+1]*nz - phi[1:nz,m-1]*nz        
     for m in range(1,nz):
         Dym[m-1,0:nx-1] = phi[m,1:nx]*nz   - phi[m-1,1:nx]*nz
         Dys[m-1,0:nx-1] = phi[m+1,1:nx]*nz - phi[m-1,1:nx]*nz 
     
     
     signum = phi_small/(np.sqrt(np.power(phi_small,2) + eps*np.power(Dxs,2) + eps*np.power(Dys,2)))
     
     psi_small   = phi_small
     
     max_it = 10
     dt = 1e-1*h/(2*np.sqrt(2)) # required by CFL condition  
     for k in range(0,max_it):
         
         psi          = np.zeros((nz+1,nx+1), dtype='float32')
         psi[1:nz,1:nx] = psi_small[0:nz-1,0:nx-1]
         
         psi[0,0:nx+1] = 2*psi[1,0:nx+1]  -psi[2,0:nx+1]
         psi[nz,0:nx+1] = 2*psi[nz-1,0:nx+1]-psi[nz-2,0:nx+1]
         
         psi[0:nz+1,0] = 2*psi[0:nz+1,1]  -psi[0:nz+1,2]
         psi[0:nz+1,nx] = 2*psi[0:nz+1,nx-1]-psi[0:nz+1,nx-2]
         
         Dxp = np.zeros((nz,nx), dtype='float32')
         Dyp = np.zeros((nz,nx), dtype='float32')
         
         Dxm = np.zeros((nz,nx), dtype='float32')
         Dym = np.zeros((nz,nx), dtype='float32')
         
         for m in range(0,nx-1):
             Dxp[0:nz-1,m]   = psi[1:nz,m+1]*nx - psi[1:nz,m]*nx
         for m in range(1,nx):
             Dxm[0:nz-1,m-1] = psi[1:nz,m]*nx   - psi[1:nz,m-1]*nx
            
         for m in range(0,nz-1):
             Dyp[m,0:nx-1]   = psi[m+1,1:nx]*nz - psi[m,1:nx]*nz
         for m in range(1,nz):
             Dym[m-1,0:nx-1] = psi[m,1:nx]*nz   - psi[m-1,1:nx]*nz
            
         Gp = np.sqrt(np.power(np.maximum(Dxm,0),2) + np.power(np.minimum(Dxp,0),2)+ np.power(np.maximum(Dym,0),2) + np.power(np.minimum(Dyp,0),2))
         Gm = np.sqrt(np.power(np.minimum(Dxm,0),2) + np.power(np.maximum(Dxp,0),2)+ np.power(np.minimum(Dym,0),2) + np.power(np.maximum(Dyp,0),2))
         
         g  = np.maximum(signum,0)*Gp + np.minimum(signum,0)*Gm
         
         #############################
         # Runge Kutta
         
         psi_small  = psi_small - dt*(g - signum)        
         
         ####################################
         # Plots
         ####################################    
         # if activeplot == 1:
         #     fig = plt.figure()
         #     ax  = fig.gca(projection='3d')
         #     plt.hold(False)
         #     surf = ax.plot_surface(XX,YY,psi_small, rstride=1, cstride=1, cmap=cm.coolwarm,
         #             linewidth=0, antialiased=False)
         #     ax.set_zlim(-1.01, 1.01)
         #     ax.zaxis.set_major_locator(LinearLocator(10))
         #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
         #     fig.colorbar(surf, shrink=0.5, aspect=5)
         #     plt.show(block=False)
         #     tm.sleep(3)
         #     plt.close()
         #
         # if activeplot == 2:
         #     fig = plt.figure()
         #     imgplot = plt.imshow(psi_small,extent = [0.0,1.0,0.0,1.0])
         #     #fig.colorbar()
         #     plt.show(block=False)
         #     tm.sleep(3)
         #     plt.close()
     return psi_small    
    
    

