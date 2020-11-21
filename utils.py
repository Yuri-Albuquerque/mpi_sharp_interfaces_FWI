import os
import errno
import sys
import shutil
import fenics as fc
from ufl import inner, grad, div, dot, dx, Measure
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from c_functions import *
from pdb import set_trace
import time
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import multiprocessing as mp
from numpy import linalg as LA

def mkDirectory(name):
    """ 
    Make a directory to save the experiments files
    """
    try:
        shutil.rmtree(name, ignore_errors=True, onerror=None)
        os.makedirs(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    print("{} : directory was created" .format(name))


def outputs_and_paths(par):
    """ 
    Create .txt experiment report, and path to source directory where .png files representing the reconstruction will be save 
    """
    mkDirectory(par["path"])
    mkDirectory(par["path"]+'MeasureField/')
    mkDirectory(par["path"]+'vel_field_plot_type_1/')
    mkDirectory(par["path"]+'vel_field_plot_type_2/')
    mkDirectory(par["path"]+'vel_field_plot_type_3/')
    mkDirectory(par["path"]+'Adjoint_Field/')
    mkDirectory(par["path"]+'shape_gradient_norm/')
    with open(par["path"]+"wave_out.txt", "w") as text_file:
        text_file.write(
            'x-axis length {} m, from start position {} m to end position {} m. \n' .format(par["xMax"] - par["xMin"], par["xMin"], par["xMax"]))
        text_file.write(
            '{} km for z-axis depth from the {} ground . \n' .format(par["zMax"], par["zMin"]))
        text_file.write('-*-Grid dimensions-*-\n')
        text_file.write(
            '    Grid Size : {:d} x {:d} and {:d} time steps\n'  .format(par["nz"], par["nx"], par["nt"]))
        text_file.write(
            'This numerical simulation only records {} seconds of wave propagation. \n' .format(par["tMax"]))
        text_file.write('The damping term only works outside of square [{}, {}] x [{}, {}] x [{}, {}] x [{}, {}]. \n'
                        .format(par["dmp_xMin"], par["zMin"], par["dmp_xMin"], par["dmp_zMax"], par["dmp_xMax"], par["dmp_zMax"], par["dmp_xMax"], par["zMin"]))
        text_file.write(
            'Damping layer width {} \n' .format(par["dmp_layer"]))
        text_file.write(
            'Discretizations steps hx = {}, hz = {} and ht = {}. \n' .format(par["hx"], par["hz"], par["ht"]))
        text_file.write('Parameters set:\n init_gu = {}\n shots amount = {}\n receivers amount = {}\n' .format(
            par["i_guess"], par["n_shots"], par["n_receivers"]))
        text_file.write('gamma = {}\n gamma2 = {}\n ls_max = {}\n' .format(
            par["gamma"], par["gamma2"], par["ls_max"]))
        text_file.write('stop_coeff = {}\n add_noise = {}\n add_weight = {}\n '.format(
            par["stop_coeff"], par["add_noise"], par["add_weight"]))
        text_file.write('beta0_init = {}\n stop_decision_limit = {}\n alpha1 = {}\n alpha2 = {}\n peak frequencies of Ricker wavelet = {}\n'.format(
            par["beta0_init"], par["stop_decision_limit"], par["alpha1"], par["alpha2"], par["source_peak_frequency"]))
        text_file.write(
            'Courant number for state/adjoint: {}\n' .format(par["CFL"]))

    with open(par["path"]+"receivers_location.txt", "a") as text_file:
        text_file.write(
            'Grid receivers indexes for the state/adjoint:\n {} ' .format(par["rec_index"]))
        text_file.write('\n'+'\n')
        text_file.write(
            'Grid receivers locations the state/adjoint:\n rec = {}' .format(par["rec"]))
# ------------------------------------------------------------------------


def stateSolution(parameters, eta_u, u, FT_state, vel):
    """ 
    Compute the state solution implemented in c_functions/src/main.cpp
    with Devito this function will no longer be useful
    """
    par = np.zeros((10))
    par[0] = parameters["xMin"]
    par[1] = parameters["xMax"]
    par[2] = parameters["zMin"]
    par[3] = parameters["zMax"]
    par[4] = parameters["tMin"]
    par[5] = parameters["tMax"]
    par[6] = parameters["hx"]
    par[7] = parameters["hz"]
    par[8] = parameters["ht"]
    return solve_wv2(par.astype("float64"), u.astype("float64"), vel.astype("float64"), eta_u.astype("float64"), FT_state.astype("float64"))
# ------------------------------------------------------------------------


def adjointSolution(parameters, eta_p, p, diff_ud, velocity):
    """
    Compute the adjoint solution implemented in c_functions/src/main.cpp
    with Devito this function will no longer be useful
    """
    nz = parameters["nz"]
    nx = parameters["nx"]
    nt = parameters["nt"]
    rec_index = parameters["rec_index"]
    # initialization of matrix solution u and the source FT
    p_tilde = np.empty((nz, nx, nt), np.dtype('float'))
    FT = np.zeros((nz, nx, nt), np.dtype('float'))
    FT_tilde = np.zeros((nz, nx, nt), np.dtype('float'))
    FT[0, rec_index, 0:nt] = 4.0*diff_ud[:,:]
    # FT_tilde = reverse_time(FT_tilde.astype("float64"), FT.astype("float64"))
    FT_tilde[0:nz, 0:nx, 0:nt] = FT[0:nz, 0:nx, ::-1]
    FT_tilde = -1.0 * FT_tilde
    #
    par = np.zeros((10))
    par[0] = parameters["xMin"]
    par[1] = parameters["xMax"]
    par[2] = parameters["zMin"]
    par[3] = parameters["zMax"]
    par[4] = parameters["tMin"]
    par[5] = parameters["tMax"]
    par[6] = parameters["hx"]
    par[7] = parameters["hz"]
    par[8] = parameters["ht"]
    #
    p_tilde = solve_wv2(par.astype("float64"), p_tilde.astype(
        "float64"), velocity.astype("float64"), eta_p.astype("float64"), FT_tilde.astype("float64"))
    # p = reverse_time(p.astype("float64"), p_tilde.astype("float64"))
    p[0:nz, 0:nx, 0:nt] = p_tilde[0:nz, 0:nx, ::-1]
    #
    del(FT, FT_tilde, p_tilde, nx, nz, nt)
    #
    return p
# -------------------------------------------------------------------------


def inside_shape(mat):
    """
    Indicator function of phi matrix
    """
    Y = np.zeros_like(mat)
    Y[mat<0] = 1.0
    return Y 
# ------------------------------------------------------------------------


def seismicAcquisitionData(synthetic_velocity_model, parameters, eta_d, d, FT_measure):
    """
    Make the synthetic seismogram
    with Devito this function will no longer be useful
    """
    par = np.zeros((10))
    par[0] = parameters["xMin"]
    par[1] = parameters["xMax"]
    par[2] = parameters["zMin"]
    par[3] = parameters["zMax"]
    par[4] = parameters["tMin"]
    par[5] = parameters["tMax"]
    par[6] = parameters["hx"]
    par[7] = parameters["hz"]
    par[8] = parameters["ht"]
    return solve_wv2(par.astype("float64"), d.astype("float64"), synthetic_velocity_model.astype("float64"), eta_d.astype("float64"), FT_measure.astype("float64"))
# ------------------------------------------------------------------------

class WaveSolver():
    def __init__(self, parameters, eta):
        self.parameters = parameters
        self.eta = eta

    def state(self, u, vel_field, FT):
        return stateSolution(self.parameters, self.eta, u, FT, vel_field)

    def adjoint(self, P, vel_field, misfit):
        return adjointSolution(self.parameters, self.eta, P, misfit, vel_field)

    def measurements(self, d, synth_model, FT, add_noise):
        nz, nx, nt = d.shape
        d = seismicAcquisitionData(
            synth_model, self.parameters, self.eta, d, FT)
        Nx = self.parameters['nx']
        Nz = self.parameters['nz']
        Nt = self.parameters['nt']
        gc_t = self.parameters['gc_t']
        rec_index = self.parameters['rec_index']
        noise_lv = self.parameters['noise_lv']
        path = self.parameters['path']
        if add_noise:
            tt = 2.0 * np.random.rand(Nz, Nx, Nt)-1.0
            d_max = np.max(np.abs(d))
            noise = noise_lv*d_max*tt
            noise_level = simpson_rule(np.sum(
                np.power(noise[0, rec_index, 0:Nt], 2), axis=0), gc_t)\
                / simpson_rule(np.sum(
                    np.power(d[0, rec_index, 0:Nt], 2), axis=0), gc_t)
            noise_level = np.sqrt(noise_level)
            SNR = 10*np.log10(1.0/noise_level)
            d += noise
            # Report noise level
            with open(path+"wave_out.txt", "a") as text_file:
                text_file.write('\n')
                text_file.write('-------------------------\n')
                text_file.write('Noise coefficient:{}\n' .format(noise_lv))
                text_file.write(
                    'Noise level: {:.2f} % \n ' .format(100*noise_level))
                text_file.write(
                    'Signal-to-Noise ratio (SNR): {:.2f} dB\n' .format(SNR))
            del(noise)

        return d


def simpson_rule(vec_f, x):
    """
    Simpson rule integrator. Entries are both 1D np.array with same length
    """
    N = len(x)+1 if len(x) % 2 == 0 else len(x)
    h = np.diff(x)
    f = np.zeros((N), np.dtype('float'))
    f[0:len(x)] = vec_f[0:len(x)]
    f[N-1] = vec_f[len(x)-1]
    c = np.zeros((N), np.dtype('float'))
    c[1:N-1:2] = 4.0
    c[2:N-1:2] = 2.0
    c[0] = 1.0
    c[N-1] = 1.0
    result = (h[0]/3.0) * np.dot(c, f)
    return result
# ------------------------------------------------------------------------


def calc_state_adjoint_derivatives(u, P, uz, pz, ux, px, ut, pt, hz, hx, ht):
    """
    Compute grad(u) and grad(P) that compose shape derivative equation
    """
    nz, nx, nt = u.shape
    ut = dt_u(u.astype("float64"), ut.astype("float64"), ht)
    pt = dt_p(P.astype("float64"), pt.astype("float64"), ht)
    ux = dx_cpp(u.astype("float64"), ux.astype("float64"), hx)
    px = dx_cpp(P.astype("float64"), px.astype("float64"), hx)
    uz = dz_cpp(u.astype("float64"), uz.astype("float64"), hz)
    pz = dz_cpp(P.astype("float64"), pz.astype("float64"), hz)
    
    # ut[0:nz, 0:nx, 1:nt] = (
    #     1.0/ht)*(u[0:nz, 0:nx, 1:nt] - u[0:nz, 0:nx, 0:nt-1])
    
    # pt[0:nz, 0:nx, 1:nt] = (
    #     1.0/ht)*(P[0:nz, 0:nx, 1:nt] - P[0:nz, 0:nx, 0:nt-1])
    #
    # ut[0:nz, 0:nx, 0] = 0.0
    # pt[0:nz, 0:nx, 0] = pt[0:nz, 0:nx, 1]
    # -----------------------------
    # differences of u_x and p_x
    # -----------------------------
    # ux[0:nz, 1:nx - 1, 0:nt] = 0.5 * (1.0/hx) * \
    #     (u[0:nz, 2:nx, 0:nt] - u[0:nz, 0:nx - 2, 0:nt])

    # ux[0:nz, 0, 0:nt] = ux[0:nz, 1, 0:nt]
    # ux[0:nz, nx - 1, 0:nt] = ux[0:nz, nx - 2, 0:nt]


    # px[0:nz, 1:nx - 1, 0:nt] = 0.5 * (1.0/hx) * \
    #     (P[0:nz, 2:nx, 0:nt] - P[0:nz, 0:nx - 2, 0:nt])
    # px[0:nz, 0, 0:nt] = px[0:nz, 1, 0:nt]
    # px[0:nz, nx - 1, 0:nt] = px[0:nz, nx - 2, 0:nt]
    # set_trace()
    # -----------------------------
    # differences of u_z and p_z
    # -----------------------------
    # uz[1:nz - 1, 0:nx, 0:nt] = 0.5 * (1.0/hz) * \
    #     (u[2:nz, 0:nx, 0:nt] - u[0:nz - 2, 0:nx, 0:nt])
    
    # uz[0, 0:nx, 0:nt] = uz[1, 0:nx, 0:nt]
    # uz[nz - 1, 0:nx, 0:nt] = uz[nz - 2, 0:nx, 0:nt]

    
    # pz[1:nz - 1, 0:nx, 0:nt] = 0.5 * (1.0/hz) * \
    #     (P[2:nz, 0:nx, 0:nt] - P[0:nz - 2, 0:nx, 0:nt])
   
    # pz[0, 0:nx, 0:nt] = pz[1, 0:nx, 0:nt]
    # pz[nz - 1, 0:nx, 0:nt] = pz[nz - 2, 0:nx, 0:nt]

    return uz, pz, ux, px, ut, pt
# ------------------------------------------------------------------------


# def calc_eta_derivatives(eta, etax, etaz, hz, hx, ht):
#     """
#     Compute grad(eta) term that compose shape derivative equation
#     """
#     nz, nx = eta.shape
#     etax[0:nz, 1:nx - 1] = 0.5 * (1.0/hx) * (eta[0:nz, 2:nx] - eta[0:nz, 0:nx - 2])
#     etax[0:nz, 0] = etax[0:nz, 1]
#     etax[0:nz, nx - 1] = etax[0:nz, nx - 2]
#     # -----------------------------
#     # forward differences for eta z-direction
#     # -----------------------------
#     etaz[1:nz - 1, 0:nx] = 0.5 * (1.0/hz) * (eta[2:nz, 0:nx] - eta[0:nz - 2, 0:nx])
#     etaz[0, 0:nx] = etaz[1, 0:nx]
#     etaz[nz - 1, 0:nx] = etaz[nz - 2, 0:nx]

#     return etaz, etax
# ------------------------------------------------------------------------


def calc_derivative_sh(u, P, eta, hz, hx, ht, V):
    """
    Compute time integration for shape derivative terms
    """

    nz, nx, nt = u.shape
    uz = np.zeros((nz, nx, nt))
    pz = np.zeros((nz, nx, nt))
    ux = np.zeros((nz, nx, nt))
    px = np.zeros((nz, nx, nt))
    ut = np.zeros((nz, nx, nt))
    pt = np.zeros((nz, nx, nt))
    # etax = np.zeros((nz, nx))
    # etaz = np.zeros((nz, nx))
    #
    simpson_coeffs = np.zeros((nt))
    simpson_coeffs[1: nt-1: 2] = 4.0
    simpson_coeffs[2: nt-1: 2] = 2.0
    simpson_coeffs[0] = 1.0
    simpson_coeffs[nt - 1] = 1.0
    simpson_coeffs = (ht/3.0) * simpson_coeffs
    #

    uz, pz, ux, px, ut, pt = calc_state_adjoint_derivatives(
        u, P, uz, pz, ux, px, ut, pt, hz, hx, ht)

    k0 = np.zeros((nz * nx))
    k0 = int_0T(k0.astype("float64"), ut.astype("float64"), pt.astype("float64"), simpson_coeffs.astype("float64"))
    
    k3_xx = np.zeros((nz * nx))
    k3_xx = int_0T(k3_xx.astype("float64"), ux.astype("float64"), px.astype("float64"), simpson_coeffs.astype("float64"))
    
    k3_xz = np.zeros((nz * nx))
    k3_xz = int_0T(k3_xz.astype("float64"), ux.astype("float64"), pz.astype("float64"), simpson_coeffs.astype("float64")) 
    k3_xz += int_0T(k3_xz.astype("float64"), uz.astype("float64"), px.astype("float64"), simpson_coeffs.astype("float64")) 
    
    k3_zz = np.zeros((nz * nx))
    k3_zz = int_0T(k3_zz.astype("float64"), uz.astype("float64"), pz.astype("float64"), simpson_coeffs.astype("float64"))

    k2 = k3_xx + k3_zz
    #etaz, etax = calc_eta_derivatives(eta, etax, etaz, hz, hx, ht)


    # utpt = ut[0:nz, 0:nx, 0:nt] * pt[0:nz, 0:nx, 0:nt]
    #k0 = np.sum(utpt*simpson_coeffs[0:nt], axis=2)
    # k0 = np.dot(utpt,simpson_coeffs)


    
    # alternative way of calculating k3
    # uxpx = ux[0:nz, 0:nx, 0:nt] * px[0:nz, 0:nx, 0:nt]
    #k3_xx = np.sum(uxpx[0:nz, 0:nx, 0:nt]*simpson_coeffs[0:nt], axis=2)
    # k3_xx = np.dot(uxpx,simpson_coeffs)


    # uxpz = ux[0:nz, 0:nx, 0:nt] * pz[0:nz, 0:nx, 0:nt]
    # uzpx = uz[0:nz, 0:nx, 0:nt] * px[0:nz, 0:nx, 0:nt]
    #k3_xz = np.sum(
    #    (uxpz[0:nz, 0:nx, 0:nt] + uzpx[0:nz, 0:nx, 0:nt])*simpson_coeffs[0:nt], axis=2)
    # k3_xz = np.dot(uxpz + uzpx,simpson_coeffs)


    # uzpz = uz[0:nz, 0:nx, 0:nt] * pz[0:nz, 0:nx, 0:nt]
    #k3_zz = np.sum(uzpz[0:nz, 0:nx, 0:nt]*simpson_coeffs[0:nt], axis=2)
    # k3_zz = np.dot(uzpz,simpson_coeffs)

    
    # calculating k2
    # k2 = k3_xx[0:nz, 0:nx] + k3_zz[0:nz, 0:nx]
    # calculating k4
    #k4 = (np.sum((ut[0:nz, 0:nx, 0:nt] * P[0:nz, 0:nx, 0:nt]) *
    #                simpson_coeffs[0:nt], axis=2))*eta[0:nz, 0:nx]

    #k4_0 = (np.sum((ut[0:nz, 0:nx, 0:nt] * P[0:nz, 0:nx, 0:nt]) *
    #                simpson_coeffs[0:nt], axis=2))
    #k4_0 = np.dot(ut*P,simpson_coeffs)


    #k4 = k4_0*eta[0:nz, 0:nx] 
    # calculating k5
    #k5_x = (np.sum((ut[0:nz, 0:nx, 0:nt] * P[0:nz, 0:nx, 0:nt]) *
    #                  simpson_coeffs[0:nt], axis=2))*etax[0:nz, 0:nx]
    #k5_z = (np.sum((ut[0:nz, 0:nx, 0:nt] * P[0:nz, 0:nx, 0:nt]) *
    #                  simpson_coeffs[0:nt], axis=2))*etaz[0:nz, 0:nx]
    #
    #k5_x = k4_0*etax[0:nz, 0:nx]
    #k5_z = k4_0*etaz[0:nz, 0:nx]

    return k0[fc.dof_to_vertex_map(V)], k2[fc.dof_to_vertex_map(V)], k3_zz[fc.dof_to_vertex_map(V)], k3_xz[fc.dof_to_vertex_map(V)], k3_xx[fc.dof_to_vertex_map(V)]
# -----------------------------------------------------------------------


def derive_sh(parameters, u, P, eta, V, csi, dx, seismic_vel, dtotal):
    """
    Assemble shape derivative equation
    """
    hx, hz, ht = parameters["hx"], parameters["hz"], parameters["ht"]

    #
    nz, nx, nt = u.shape
    u_fe = fc.Function(V)
    p_fe = fc.Function(V)
    k0_fe = fc.Function(V)
    k2_fe = fc.Function(V)
    k3_xx_fe = fc.Function(V)
    k3_xz_fe = fc.Function(V)
    k3_zz_fe = fc.Function(V)
    #k4_fe = fc.Function(V)
    #k5_x_fe = fc.Function(V)
    #k5_z_fe = fc.Function(V)
    #
    # k0_fe.vector()[:] = k0.reshape((nx*nz))[fc.dof_to_vertex_map(V)]
    # k3_xx_fe.vector()[:] = k3_xx.reshape((nx * nz))[fc.dof_to_vertex_map(V)]
    # k3_xz_fe.vector()[:] = k3_xz.reshape((nx * nz))[fc.dof_to_vertex_map(V)]
    # k3_zz_fe.vector()[:] = k3_zz.reshape((nx * nz))[fc.dof_to_vertex_map(V)]
    # k2_fe.vector()[:] = k2.reshape((nx * nz))[fc.dof_to_vertex_map(V)]

    k0_fe.vector()[:], k2_fe.vector()[:], k3_zz_fe.vector()[:], k3_xz_fe.vector()[:], k3_xx_fe.vector()[:] = calc_derivative_sh(
        u, P, eta, hz, hx, ht, V)
      
    #k4_fe.vector()[:] = k4.reshape((nx * nz))[fc.dof_to_vertex_map(V)]
    #k5_x_fe.vector()[:] = k5_x.reshape((nx * nz))[fc.dof_to_vertex_map(V)]
    #k5_z_fe.vector()[:] = k5_z.reshape((nx * nz))[fc.dof_to_vertex_map(V)]
    #
    sigma0 = 1.0/(seismic_vel[1]*seismic_vel[1])
    sigma1 = 1.0/(seismic_vel[0]*seismic_vel[0])
    #
    rhs = -1.0*(sigma0 * k0_fe * div(csi) * dx(0) +
                sigma1 * k0_fe * div(csi) * dx(1))
    rhs += 1.0 * (k2_fe * div(csi) * dtotal)
    #rhs += 1.0 * ((k2_fe + k4_fe) * div(csi) * dtotal)
    rhs += -1.0*((2.0*k3_zz_fe*grad(csi)[1, 1] + k3_xz_fe * (
        grad(csi)[0, 1] + grad(csi)[1, 0]) + 2.0*k3_xx_fe*grad(csi)[0, 0]) * dtotal)
    #rhs += 1.0 * (k4_fe * div(csi)*dtotal)

    #rhs += 1.0 * ((k5_x_fe * csi[0] + k5_z_fe*csi[1]) * dtotal)
    #
    return rhs


# ------------------------------------------------------------------------
#


def source(tme, peak_frequency):
    """ 
    The amplitude A of the Ricker wavelet
    """
    pk_f = peak_frequency  # (hertz) peak frequency
    def a(t): return np.pi*(pk_f*2.0*t - 0.95)
    def s(t): return 4.0*1.0e3*(1.0 - 2.0*(a(t)*a(t)))*np.exp(-(a(t)*a(t)))
    return s(tme)
# ------------------------------------------------------------------------


def source_config(parameters):
    """
    Compute source matrix to build the seismograms of synthetic model
    with Devito this function will no longer be useful
    """
    hx = parameters["hx"]
    nx = parameters["nx"]
    nz = parameters["nz"]
    xMin = parameters["xMin"]
    nt = parameters["nt"]
    grid_coords_t = parameters["gc_t"]
    grid_x_m = parameters["gc_x"]
    src_Zpos = parameters["src_Zpos"]
    dmp_xMin = parameters["dmp_xMin"]
    dmp_xMax = parameters["dmp_xMax"]
    strikes = parameters["n_shots"]
    peak_frequency = parameters["source_peak_frequency"]
    FT = np.zeros((strikes, nz, nx, nt), np.dtype('float'))
    IS = np.int(np.floor(src_Zpos))
    JS_pos = np.zeros((strikes), np.dtype('int'))
    path = parameters["path"]
    text_file = open(path+"wave_out.txt", "a")
    if strikes == 1:
        text_file.write('\n')
        src_Xpos = 0.5*(dmp_xMin + dmp_xMax)
        text_file.write(
            'One source located at: (0.0, {}) \n' .format(src_Xpos))
        JS_pos[0] = int(round(abs(
            src_Xpos - xMin)/hx))
        FT[0, IS, JS_pos[0], 0:nt] = source(grid_coords_t, peak_frequency)
    else:
        # computing source position in x-axis
        JS_pos = np.linspace(
            parameters["id_dmp_xMin"]+1, parameters["id_dmp_xMax"]-1, strikes, dtype='int')
        source_distr = grid_x_m[JS_pos]
        ricker = np.copy(source(grid_coords_t, peak_frequency))
        for i, js in enumerate(JS_pos):
            FT[i, IS, js, 0:nt] = ricker
    text_file.write('Signal sources located at \n {}\n' .format(source_distr))
    text_file.write('Signal source matrix indexes \n {}\n' .format(JS_pos))
    text_file.close()
    plt.figure()
    plt.plot(grid_coords_t,FT[1,IS,JS_pos[1],:])
    plt.savefig(str(path + 'source.png'), dpi=500)
    return FT
# ------------------------------------------------------------------------


def dmp(parameters):
    """
    Compute damping nz x nx matrix to calculate state and seismograms
    with Devito this function will no longer be necessarily 
    """
    #
    par = np.zeros((12))
    par[0] = parameters["xMin"]
    par[1] = parameters["xMax"]
    par[2] = parameters["zMin"]
    par[3] = parameters["zMax"]
    par[4] = parameters["tMin"]
    par[5] = parameters["tMax"]
    nz = parameters["nz"]
    nx = parameters["nx"]
    par[6] = parameters["hx"]
    par[7] = parameters["hz"]
    par[8] = parameters["ht"]
    par[9] = parameters["dmp_xMin"]
    par[10] = parameters["dmp_xMax"]
    par[11] = parameters["dmp_zMax"]
    #
    eta_mat = np.zeros((nz, nx), np.dtype('float'))
    eta = damping_function(eta_mat.astype("float64"),
                           par.astype("float64"))
    
    gnu_data(eta, 'damp.dat', parameters)
    
    return eta
# ------------------------------------------------------------------------


def initial_guess(parameters):
    """
    Compute phi matrix function that will be the initial guess in 1st iteration 
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    init_gu = parameters["i_guess"]
    nz = parameters["nz"]
    nx = parameters["nx"]
    xList = np.linspace(xMin, xMax, nx)
    zList = np.linspace(zMin, zMax, nz)
    XX, ZZ = np.meshgrid(xList, zList)
    if init_gu == 1:
        x0 = 0.500
        x1 = 0.300
        phi_mat = np.sqrt(np.square((XX - x0)/2.0) +
                          np.square((ZZ - x1)/1.0)) - 0.12
    elif init_gu == 2:
        x00 = 0.300
        x01 = 0.310
        x10 = 0.700
        x11 = 0.210
        phi_mat = np.minimum((np.sqrt(np.square(XX-x00) + np.square(ZZ-x01)) - 0.075),
                             (np.sqrt(np.square(XX-x10) + np.square(ZZ-x11)) - 0.075))
    elif init_gu == 3:
        x0 = 0.500
        x1 = 0.250
        phi_mat = np.sqrt(np.square((XX - x0)/1.5) +
                          np.square((ZZ - x1))) - 0.12
    elif init_gu == 4:
        x0 = 0.500
        x1 = 0.350
        phi_mat = np.sqrt(np.square((XX - x0)/2.0) +
                          np.square((ZZ - x1)/1.0)) - 0.100
    elif init_gu == 5:
        x00 = 0.350
        x01 = 0.310
        x10 = 0.630
        x11 = 0.260
        phi_mat = np.minimum(np.sqrt(np.square((XX - x00)/1.) + np.square((ZZ - x01)/1.5)) - 0.100,
                             np.sqrt(np.square((XX - x10)/1.) + np.square((ZZ - x11))) - 0.100)
    
    elif init_gu == 6:
        x00 = 0.200
        x01 = 0.250
        x10 = 0.700
        x11 = 0.210
        x20 = 0.500
        x21 = 0.110
        phi_mat = np.minimum(np.minimum((np.sqrt(np.square(XX-x00) + np.square(ZZ-x01)) - 0.07),
                             (np.sqrt(np.square(XX-x10) + np.square(ZZ-x11)) - 0.07)),
                                (np.sqrt(np.square(XX-x20) + np.square(ZZ-x21)) - 0.07))

    elif init_gu == 7:
        x00 = 0.200
        x01 = 0.250
        x10 = 0.700
        x11 = 0.210
        x20 = 0.500
        x21 = 0.110
        phi_mat = np.minimum(np.minimum((np.sqrt(np.square(XX-x00) + np.square(ZZ-x01)) - 0.07),
                             (np.sqrt(np.square(XX-x10) + np.square(ZZ-x11)) - 0.07)),
                                (np.sqrt(np.square(XX-x20) + np.square(ZZ-x21)) - 0.07))

    gnu_data(phi_mat, 'initial_guess.dat', parameters)
    return phi_mat
#------------------------------------------------------------------------


def plotMeasurements(parameters, d_noise):
    """
    Plot seismograms
    """
    path = parameters["path"]
    shots = parameters["n_shots"]
    nt = parameters["nt"]
    nx_m = parameters["nx"]
    times = parameters["gc_t"]
    receivers = parameters["rec_index"]
    rec_position = parameters["rec"]
    figSequence4 = 0
    print('Plotting Measure field\n')
    for s in range(shots):
        figSequence4 += 1
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 18.0))
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        ax.tick_params(axis='both', labelsize=20, width=2)
        for id_r, r in enumerate(receivers):
            signal = np.transpose(np.array(rec_position[id_r]+d_noise[s, 0, r, 0:nt], np.dtype('float')))
            ax.plot(signal, times, color='k', linewidth=0.75)
            ax.fill_betweenx(
                times, rec_position[id_r], signal, facecolor='blue', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.ylim(times[len(times)-1], times[0])
        plt.xlabel('Receiver position')
        plt.ylabel('time')
        #
        plt.title('Seismogram')
        nomeDaFigura1 = '%03d_Noise_M_Field' % (figSequence4)
        fig.savefig(str(path + 'MeasureField/' + nomeDaFigura1), dpi=500)
        plt.close()
# ------------------------------------------------------------------------


def plotstate(parameters, u, folder):
    """
    Plot state shots
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    tMin = parameters["tMin"]
    tMax = parameters["tMax"]
    path = parameters["path"]
    shots = parameters["n_shots"]
    nz = parameters["nz"]
    nx = parameters["nx"]
    nt = parameters["nt"]
    times = parameters["gc_t"]
    receivers = parameters["rec_index"]
    rec_position = parameters["rec"]
    # plot state solution
    figSequence2 = 0
    print('Plotting wave field for the initial velocity guess...\n')
    for s in range(shots):
        plt.close()
        figSequence2 += 1
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 18.0))
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        ax.tick_params(axis='both', labelsize=20, width=2)
        for id_r, r in enumerate(receivers):
            signal = np.transpose(
                np.array(rec_position[id_r]+u[s, 0, r, 0:nt], np.dtype('float')))
            ax.plot(signal, times, color='k', linewidth=0.75)
            ax.fill_betweenx(
                times, rec_position[id_r], signal, facecolor='green', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.ylim(times[len(times)-1], times[0])
        plt.xlabel('Receiver position')
        plt.ylabel('time')
        nomeDaFigura2 = '%03d_shot_state_field' % (figSequence2)
        fig.savefig(str(path + folder + nomeDaFigura2), dpi=fig.dpi)
        plt.close()
# ------------------------------------------------------------------------
def plot_displacement_field(parameters, mat):
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path"]
    nz, nx, nt = mat.shape
    figSequence3 = 0
    mkDirectory(parameters["path"]+'./displacement')
    print('Plotting displacement field...\n')
    # XX, ZZ = np.meshgrid(np.linspace(xMin, xMax, nx),np.linspace(zMin, zMax, nz))
    for t in range(nt):
        plt.close()
        if t % 20 == 0:
            figSequence3 += 1
            fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
            plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
            plt.imshow(np.array(mat[0:nz, 0:nx, t], np.dtype(
                'float')),  cmap='gray', origin='upper', extent=[xMin, xMax, zMax, zMin])
            plt.xlabel('$x$')
            plt.ylabel('$z$')
            ax.tick_params(axis='both', labelsize=20, width=2)
            nomeDaFigura3 = '%03d_displacement_field' % (figSequence3)
            fig.savefig(str(path + 'displacement/' + nomeDaFigura3),
                        dpi=fig.dpi, bbox_inches='tight')
            plt.close()


def plotadjoint(parameters, p):
    """
    Plot adjoint  
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path"]
    nx = parameters["nx"]
    nz = parameters["nz"]
    nt = parameters["nt"]
    figSequence3 = 0
    print('Plotting Adjoint field...\n')
    xx = np.linspace(xMin, xMax, nx)
    zz = np.linspace(zMin, zMax, nz)
    XX, ZZ = np.meshgrid(xx, zz)
    for plotTimes3 in range(nt):
        plt.close()
        if plotTimes3 % 20 == 0:
            figSequence3 += 1
            fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
            # ax = fig.gca(projection='3d')
            # plt.title('Adjoint field')
            # surf = ax.plot_surface(XX,ZZ,np.array(p[0:nz,0:nx,plotTimes3], np.dtype('float')), rstride=1, cstride=1,\
            # cmap=cm.bone,linewidth=0,antialiased=False)
            # ax.set_zlim(-0.01, 0.01)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
            plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
            # title = 'Adjoint'
            # plt.title(str(title)+'\n')
            plt.imshow(np.array(p[0:nz, 0:nx, plotTimes3], np.dtype(
                'float')),  cmap='gray', origin='upper', extent=[xMin, xMax, zMax, zMin])
            #  plt.axis([xMin, xMax, zMax, zMin])
            plt.xlabel('$x$')
            plt.ylabel('$z$')
            ax.tick_params(axis='both', labelsize=20, width=2)
            #  plt.gca().grid(color='gray', linestyle=':', linewidth=0.8)
            #  cbaxes = fig.add_axes([0.93, 0.35, 0.02, 0.3])
            #  fig.colorbar(surf, shrink=0.5, aspect=5, label='Amplitude', cax = cbaxes)
            nomeDaFigura3 = '%03d_A_field' % (figSequence3)
            fig.savefig(str(path + 'Adjoint_Field/' + nomeDaFigura3),
                        dpi=fig.dpi, bbox_inches='tight')
            plt.close()
# ------------------------------------------------------------------------
def plot_misfit(parameters, file_name, graph_title, mat):
    path = parameters["path_misfit"]
    times = parameters["gc_t"]
    rec_position = parameters["rec"]
    shots, n_receivers, nt = mat.shape 
    count = 0
    #
    for s in range(shots):
        plt.close()
        count += 1
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 18.0))
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        ax.tick_params(axis='both', labelsize=20, width=2)
        for id_r, r in enumerate(rec_position):
            signal = np.transpose(np.array(r+mat[s,id_r, 0:nt], np.dtype('float')))
            ax.plot(signal, times, color='k', linewidth=0.75)
            ax.fill_betweenx(
                times, rec_position[id_r], signal, facecolor='black', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.ylim(times[len(times)-1], times[0])
        plt.title(graph_title)
        plt.xlabel('position')
        plt.ylabel('time')
        name = '%03d_' % (count)
        name += file_name
        fig.savefig(str(path + name + '.png'), dpi=fig.dpi)
        plt.close()

def plot_mat(parameters, file_name, graph_title, mat):
    """
    Generic imshow of 2D matrix 
    """
    path = parameters["path"]
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    cs = ax.imshow(np.array(mat, np.dtype('float')),  cmap='ocean_r',
                   origin='upper')
    ax.tick_params(axis='both', labelsize=20, width=2)
    fig.colorbar(cs, shrink=0.35, aspect=5)
    fig.savefig(str(path + str(file_name))+'.png',
                dpi=fig.dpi, bbox_inches='tight')
    plt.close()
# ------------------------------------------------------------------------


def plot_countour(parameters, name, mat, cont):
    """
    Plot 0 level of phi matrix function
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path_phi"]
    nz = parameters["nz"]
    nx = parameters["nx"]
    xlist = np.linspace(xMin, xMax, nx)
    zlist = np.linspace(zMin, zMax, nz)
    Xx, Zz = np.meshgrid(xlist, zlist)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.title('')
    plt.contour(Xx, Zz, mat, [0.0], extent=[
        xMin, xMax, zMax, zMin], colors='red', origin='upper', linestyles='dotted', linewidths=3.0)
    ax.tick_params(axis='both', labelsize=20, width=2)
    plt.title('$\phi$ $0$ level contour')
    plt.gca().invert_yaxis()
    fig_ind = '%03d_' % (cont)
    fig.savefig(str(path+fig_ind+name+'.png'),
                format='png',  dpi=300, bbox_inches='tight')
    plt.close()


def plot_mat3D(parameters, file_name, mat, cont):
    """
    Plot 3D phi
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path_phi"]
    nz = parameters["nz"]
    nx = parameters["nx"]
    xx = np.linspace(xMin, xMax, nx)
    zz = np.linspace(zMin, zMax, nz)
    XX, ZZ = np.meshgrid(xx, zz)
    plt.close()
    fig = plt.figure(figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    ax = fig.add_subplot(111, projection='3d')
    surf1 = ax.plot_surface(XX, ZZ, mat, rstride=1, cstride=1,
                            cmap='viridis_r', linewidth=0, alpha=0.95, antialiased=True)
    plt.title('3d $\phi$')
    ax.set_zlim(np.min(mat), np.max(mat))
    ax.set_ylim(zMax, zMin)
    ax.view_init(elev=30., azim=255)
    plt.xlabel('length')
    plt.ylabel('depth')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    cbaxes = fig.add_axes([0.88, 0.35, 0.02, 0.3])
    fig.colorbar(surf1, shrink=0.35, aspect=5, cax=cbaxes)
    fig_ind = '%03d_' % (cont)
    plt.savefig(str(path + fig_ind + file_name + '.png'),
                dpi=fig.dpi, bbox_inches='tight')
    plt.close()
# ------------------------------------------------------------------------


def plottype1(parameters, mat, matcontour, cont):
    """
    Plot superposition of ground truth and reconstruction contour
    """ 
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path"]
    nz = parameters["nz"]
    nx = parameters["nx"]
    xlist = np.linspace(xMin, xMax, nx)
    zlist = np.linspace(zMin, zMax, nz)
    Xx, Zz = np.meshgrid(xlist, zlist)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    greys = cm.get_cmap('Greys')
    newcolors = ListedColormap(greys(range(130)))
    plt.imshow(mat[:, :], cmap=newcolors,
               extent=[xMin, xMax, zMax, zMin], interpolation='none')
    plt.contour(Xx, Zz, matcontour, [0.0], extent=[
        xMin, xMax, zMax, zMin], colors='red', origin='upper', linestyles='dotted', linewidths=3.0)
    ax.tick_params(axis='both', labelsize=20, width=2)
    nomeDaFigura2 = '%03d_vel_field' % (cont)
    fig.savefig(str(path+'vel_field_plot_type_1/' + nomeDaFigura2+'.png'), format='png',
                dpi=300, bbox_inches='tight')
    plt.close()
    del(newcolors)
# ------------------------------------------------------------------------


def plottype2(parameters, mat1, mat2, cont):
    """
    Plot superposition of ground truth and reconstruction matrix
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path"]
    phi_mat_ones = inside_shape(mat2)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    greys = cm.get_cmap('Greys')
    greys = greys(range(130))
    greys[0] = [0.0, 0.0, 0.0, 0.0]
    newcolors = ListedColormap(greys)
    reds = cm.get_cmap('Reds')
    reds = reds(range(200))
    reds[0] = [0.0, 0.0, 0.0, 0.0]
    newcolors_2 = ListedColormap(reds)
    plt.imshow(mat1[:, :], cmap=newcolors,
               extent=[xMin, xMax, zMax, zMin], interpolation='none')
    plt.imshow(phi_mat_ones, extent=[
        xMin, xMax, zMax, zMin], cmap=newcolors_2, origin='upper', alpha=0.6)
    ax.tick_params(axis='both', labelsize=20, width=2)
    nomeDaFigura2 = '%03d_vel_field' % (cont)
    fig.savefig(str(path+'vel_field_plot_type_2/' + nomeDaFigura2 +
                    '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    del(newcolors)
# ------------------------------------------------------------------------


def plottype3(parameters, mat1, mat2, mainIt, cont):
    """
    Plot ground truth and reconstruction in different images 
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path"]
    phi_mat_ones = inside_shape(mat2)
    if mainIt == 0:
        plt.close()
        fig, ax = plt.subplots(1, 1, num=0, figsize=(9.0, 7.0))
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        greys = cm.get_cmap('Greys')
        newcolors = ListedColormap(greys(range(130)))
        plt.imshow(
            mat1[:, :], extent=[xMin, xMax, zMax, zMin], cmap=newcolors)
        ax.tick_params(axis='both', labelsize=20, width=2)
        nomeDaFigura2 = '%03d_groud_truth' % (cont)
        fig.savefig(str(path+'vel_field_plot_type_3/' + nomeDaFigura2 +
                        '.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
        del(newcolors)
    # ------------------------
    plt.close()
    fig, ax = plt.subplots(1, 1, num=1, figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    get_map_colors = cm.get_cmap('tab20c')
    colors_vec = get_map_colors(range(4, 8, 1))
    colors_vec = colors_vec[::-1]
    colors_vec[0] = [0.0, 0.0, 0.0, 0.0]
    newcolors = ListedColormap(colors_vec)
    plt.imshow(phi_mat_ones[:, :], extent=[xMin,
                                           xMax, zMax, zMin], cmap=newcolors, origin='upper')
    ax.tick_params(axis='both', labelsize=20, width=2)
    nomeDaFigura2 = '%03d_vel_field' % (cont)
    fig.savefig(str(path+'vel_field_plot_type_3/' + nomeDaFigura2 +
                    '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    del(newcolors)
# ------------------------------------------------------------------------


def plottype4(parameters, mat, cont):
    """
    Plot the difference between ground truth and reconstruction
    """
    xMin = parameters["xMin"]
    xMax = parameters["xMax"]
    zMin = parameters["zMin"]
    zMax = parameters["zMax"]
    path = parameters["path"]
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    get_map_colors = cm.get_cmap('tab20c')
    colors_vec = get_map_colors(range(17))
    colors_vec[0] = [0.0, 0.0, 0.0, 0.0]
    newcolors = ListedColormap(colors_vec)
    plt.imshow(mat[:, :], extent=[xMin, xMax, zMax, zMin], cmap=newcolors)
    ax.tick_params(axis='both', labelsize=20, width=2)
    nomeDaFigura2 = '%03d_diff' % (cont)
    fig.savefig(str(path+'vel_field_plot_type_4/' + nomeDaFigura2 +
                    '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    del(newcolors)
# ------------------------------------------------------------------------


def plotcostfunction(parameters, vec, mainIteff):
    if mainIteff >1 :
        path = parameters["path"]
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
        plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
        plt.plot(np.arange(0, mainIteff, 1), np.log10(
            vec[0:mainIteff]), color='red', linestyle='-', linewidth=1.0)
        # plt.title("Cost Function $J(\Omega)$ in $\log_{10}$ scale")
        plt.xlim(0, mainIteff)
        plt.xlabel('Iteration')
        plt.ylabel('$\log_{10}\; J(\Omega)$')
        ax.tick_params(axis='both', labelsize=20, width=2)
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.gca().grid(color='black', linestyle=':', linewidth=0.8)
        plt.savefig(path + 'cost_history.png',
                    dpi=fig.dpi, bbox_inches='tight')
        plt.close()
    else:
        return None
# ------------------------------------------------------------------------


def plotnormtheta(parameters, vec, mainIteff):
    if mainIteff > 1:
        path = parameters["path"]
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))
        plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
        plt.plot(np.arange(0, mainIteff, 1),
                vec[0:mainIteff], color='blue', linestyle='-', linewidth=1.0)
        # plt.title('Norm of descent gradient $ \\theta $ ')
        plt.xlim(0, mainIteff)
        plt.xlabel('Iteration')
        ax.tick_params(axis='both', labelsize=20, width=2)
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.gca().grid(color='black', linestyle=':', linewidth=0.8)
        plt.savefig(path + 'theta_history.png',
                    dpi=fig.dpi, bbox_inches='tight')
    else:
        return None

def gnu_data(mat, file_name, par):
    """
    save 2D numpy array in gnuplot format
    Entries: 
    mat, is a 2D numpy array
    file_name, is string that ends with .dat naming file
    par, parameters dictionary 
    """
    xMin = par["xMin"]
    xMax = par["xMax"]
    zMin = par["zMin"]
    zMax = par["zMax"]
    path = par["path"]
    nz, nx = mat.shape
    xlist = np.linspace(xMin, xMax, nx)
    zlist = np.linspace(zMin, zMax, nz)
    XX, ZZ = np.meshgrid(xlist, zlist)
    data_to_text = np.zeros((nz*nx, 3))
    data_to_text[:, 0] = np.reshape(XX, nz*nx)
    data_to_text[:, 1] = np.reshape(ZZ, nz*nx)
    data_to_text[:, 2] = np.round(np.reshape(mat, nz*nx), 8)
    np.savetxt(path+ file_name, data_to_text, fmt="%s", delimiter=' ')
    
    indexes = np.arange(nx, nx*nz, nx)
    with open(path+file_name, "r") as f:
        contents = f.readlines()
    
    for item, offset in enumerate(indexes):
        contents.insert(item+offset, "\n")
    
    with open(path+file_name, "w") as f:
        contents = "".join(contents)
        f.write(contents)
    
    return None
