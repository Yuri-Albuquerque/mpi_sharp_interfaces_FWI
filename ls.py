import numpy
import math
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.image as mpimg
import time as tm
from c_functions import hj
from utils import plot_mat, plot_mat3D

def hamiltonjacobi(v1, v2, phiraw, parameters, beta, MainItEff) :
     #-----------------
     # Data
     #-----------------
     #itermax = 6 # before it was 10
     itermax = 10 

     # pdb.set_trace()
     norm_v = np.sqrt(v1**2 +  v2**2)

     par = np.zeros(4)
     par[0] = beta
     par[1] = itermax
     par[2] = parameters['hz']
     par[3] = parameters['hx']
     #------------------------------------
     # Plotting options
     #------------------------------------
     active_plot = 1
     if active_plot == 1 and MainItEff % 10 == 0 :
         file_name = 'shape_gradient_norm/'+'%03d_shape_gradient' % (MainItEff)
         plot_mat(parameters, file_name, str('Shape gradient '+'iteration {}' .format(MainItEff)), norm_v)
         # plot_mat3D(parameters, 'shape_gradient_norm_3D', norm_v)
     
     return hj(v1.astype("float64"), v2.astype("float64"), phiraw.astype("float64"), par.astype("float64"))




