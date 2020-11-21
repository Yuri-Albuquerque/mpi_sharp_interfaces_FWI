from utils import *
from reinitialization import reinit
from ls import hamiltonjacobi
import sys
# import errno
# import os
from pdb import set_trace
# fenics-dolfin libraries
from ufl import inner, grad, div, dot, dx, Measure
import fenics as fc
# matplotlib and numpy libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from matplotlib import use
use('Agg')

sys.path.append(".")
fc.LogLevel.ERROR


def main(argv):
    data1 = {'shape01': './ground_truth/shape01.png', 'shape02': './ground_truth/shape02.png',
             'shape03': './ground_truth/shape03.png', 'shape04': './ground_truth/shape04.png', 'shape05': './ground_truth/shape05.png'}
    data2 = {'shape01': './ground_truth/200_130/shape01.png', 'shape02': './ground_truth/200_130/shape02.png',
             'shape03': './ground_truth/200_130/shape03.png', 'shape04': './ground_truth/200_130/shape04.png', 'shape05': './ground_truth/200_130/shape05.png', 'shape06': './ground_truth/200_130/shape06.png', 'shape07': './ground_truth/200_130/shape07.png'}
    data3 = {'shape01': './ground_truth/300_195/shape01.png', 'shape02': './ground_truth/300_195/shape02.png',
             'shape03': './ground_truth/300_195/shape03.png', 'shape04': './ground_truth/300_195/shape04.png', 'shape05': './ground_truth/300_195/shape05.png'}

    shots = 5
    n_rec = 80
    noise_coeff = 0.001
    gt = data1["shape01"]
    guess = 1
    test = [gt, guess, shots, n_rec, noise_coeff, str(
        "./result/{}_b/" .format(guess))]
    fwi_si(*test)


def fwi_si(gt_data, i_guess, n_shots, n_receivers, noise_lv, path):
    """
    This is the main function of the project.
    Entries 
        gt_data: string path to the ground truth image data
        i_guess: integer pointing the algorithm initialization guess
        n_shots: integer, number of strikes for the FWI
        n_receivers: integer, number of receivers for the FWI
        noise_lv: float type variable that we use to compute noise level
        path: string type variable, path to local results directory
    """
    seism_vel = [4.12, 1.95]
    image_phi = mpimg.imread(gt_data)
    chi0 = np.int64(image_phi == 0)
    chi1 = 1.0 - chi0
    synth_model = seism_vel[0]*chi1 + seism_vel[1]*chi0
    #scale in meter
    xMin = 0.0
    xMax = 1.0
    zMin = 0.0
    zMax = 0.650
    #scale in milliseconds
    tMin = 0.0
    tMax = 2.0

    # Damping layer width and damping limits
    damp_layer = 0.1*xMax
    dmp_xMin = xMin + damp_layer
    dmp_xMax = xMax - damp_layer
    dmp_zMax = zMax - damp_layer

    #    Number of grid points are determined by the loaded image size
    #    Nz, Nx are (#) of grid point
    Nz, Nx = synth_model.shape
    delta_x = xMax / Nx
    delta_z = zMax / Nz
    CFL = 0.4
    # delta_t = CFL / (max(seism_vel)*(1.0/hx_m + 1.0/hz_m))
    delta_t = (CFL * min(delta_x, delta_z)) / max(seism_vel)
    gc_t = np.arange(tMin, tMax, delta_t)
    Nt = len(gc_t)

    # Level set parameters
    MainItMax = 5000
    gamma = 0.8
    gamma2 = 0.8
    stop_coeff = 1.0e-8
    add_weight = True
    ls_max = 3
    ls = 0
    beta0_init = 1.5  # 1.2 #0.8 #0.5 #0.3
    beta0 = beta0_init
    beta = beta0
    stop_decision_limit = 150
    stop_decision = 0
    alpha1 = 0.01
    alpha2 = 0.97

    # wave Parameters
    wannaPlotFields = True
    add_noise = False if noise_lv == 0 else True
    src_Zpos = 0.0
    source_peak_frequency = 5.0  # (kilo hertz)

    # Grid coordinates
    gc_x = np.arange(xMin, xMax, delta_x)
    gc_z = np.arange(zMin, zMax, delta_z)

    # Compute receivers
    id_dmp_xMin = np.where(gc_x == dmp_xMin)[0][0]
    id_dmp_xMax = np.where(gc_x == dmp_xMax)[0][0]
    id_dmp_zMax = np.where(gc_z == dmp_zMax)[0][0]
    rec_index = np.linspace(id_dmp_xMin, id_dmp_xMax,
                            n_receivers+1, dtype='int')
    try:
        assert(len(rec_index) < id_dmp_xMax - id_dmp_xMin)
    except:
        "receivers in different positions"

    # Build the HUGE parameter dictionary
    parameters = {"gamma": gamma, 
                  "gamma2": gamma2,
                  "ls_max": ls_max, 
                  "stop_coeff": stop_coeff,
                  "add_noise": add_noise, 
                  "add_weight": add_weight,
                  "beta0_init": beta0_init, 
                  "stop_decision_limit": stop_decision_limit,
                  "alpha1": alpha1, 
                  "alpha2": alpha2,
                  "CFL": CFL, 
                  "source_peak_frequency": source_peak_frequency,
                  "src_Zpos": src_Zpos,  
                  "i_guess": i_guess,
                  "path": path, 
                  "n_shots": n_shots,
                  "n_receivers": n_receivers,
                  "nz": Nz, 
                  "nx": Nx, 
                  "nt": Nt,
                  "gc_t": gc_t, 
                  "gc_x": gc_x, 
                  "gc_z": gc_z,
                  "add_weight": add_weight, 
                  "source_peak_frequency": source_peak_frequency,
                  "xMin": xMin, 
                  "xMax": xMax, 
                  "zMin": zMin, 
                  "zMax": zMax, 
                  "tMin": tMin, 
                  "tMax": tMax,
                  "hz": delta_z, 
                  "hx": delta_x, 
                  "ht": delta_t,
                  "dmp_xMin": dmp_xMin, 
                  "dmp_xMax": dmp_xMax,
                  "dmp_zMax": dmp_zMax, 
                  "dmp_layer": damp_layer,
                  "id_dmp_xMin": id_dmp_xMin, 
                  "id_dmp_xMax": id_dmp_xMax,
                  "id_dmp_zMax": id_dmp_zMax,  
                  "rec": gc_x[rec_index],
                  "rec_index": rec_index, 
                  "noise_lv": noise_lv,
                  "path_phi": path+'phi/'}

    # Compute initial guess matrix
    outputs_and_paths(parameters)
    gnu_data(image_phi, 'ground_truth.dat', parameters)
    mkDirectory(parameters["path_phi"])
    phi_mat = initial_guess(parameters)
    ind = inside_shape(phi_mat)
    ind_c = np.ones_like(phi_mat) - ind
    vel_field = seism_vel[0] * ind + seism_vel[1] * ind_c

    # Initialization of Fenics-Dolfin functions
    # ----------------------------------------
    # Define mesh for the entire domain Omega
    # ----------------------------------------
    mesh = fc.RectangleMesh(fc.Point(xMin, zMin),
                            fc.Point(xMax, zMax), Nx-1, Nz-1)
    # ----------------------------------------
    # Function spaces
    # ----------------------------------------
    V = fc.FunctionSpace(mesh, "Lagrange", 1)
    # ----------------------------------------
    # Define operator for speed vector theta
    # ---------------------------------------
    VF = fc.VectorFunctionSpace(mesh, "Lagrange", 1)
    theta = fc.TrialFunction(VF)
    csi = fc.TestFunction(VF)
    # ----------------------------------------
    # Define boundaries of the domain
    # ----------------------------------------
    tol = fc.DOLFIN_EPS   # tolerance for coordinate comparisons

    class Left(fc.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]-xMin) < tol

    class Right(fc.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]-xMax) < tol

    class Bottom(fc.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]-zMin) < tol

    class Top(fc.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]-zMax) < tol
    # --------------------------------------
    # Initialize sub-domain instances
    # --------------------------------------
    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    # ----------------------------------------------
    # Initialize mesh function for boundary domains
    # ----------------------------------------------
    boundaries = fc.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    domains = fc.MeshFunction("size_t", mesh, mesh.topology().dim())
    left.mark(boundaries, 3)
    top.mark(boundaries, 4)
    right.mark(boundaries, 5)
    bottom.mark(boundaries, 6)
    # ---------------------------------------
    # Define operator for speed vector theta
    # ---------------------------------------
    dtotal = Measure("dx")
    dircond = 1
    # ---------------------------------------
    # setting shape derivative weights
    # ---------------------------------------
    # re-balancing sensibility to be greater at the bottom
    wei_a = (zMax + 1.0e1)**2
    wei_b = 1.0e5/zMax**2
    A_expr = str(wei_a)
    B_expr = str(wei_b)
    wei_equation = '1.0e8*(pow(x[0] - 0.5, 16) + pow(x[1] - 0.325, 10))+100'

    wei = fc.Expression(str(wei_equation), degree=1)

    # Building the left hand side of the bi-linear system
    # to obtain the descendant direction from shape derivative
    if dircond < 4:
        bcF = [fc.DirichletBC(VF, (0, 0), boundaries, 3),
               fc.DirichletBC(VF, (0, 0), boundaries, 4),
               fc.DirichletBC(VF, (0, 0), boundaries, 5),
               fc.DirichletBC(VF, (0, 0), boundaries, 6)]
    if dircond == 1:
        a = wei*alpha1 * inner(grad(theta), grad(csi)) * \
            dtotal + alpha2*wei * inner(theta, csi) * dtotal
    #
    elif dircond == 2:
        a = alpha1 * inner(grad(theta), grad(csi)) * \
            dtotal + alpha2 * inner(theta, csi) * dtotal
    elif dircond == 3:
        a = inner(grad(theta), grad(csi)) * dtotal
    elif dircond == 5:
        a = inner(grad(theta), grad(csi)) * dtotal + inner(theta, csi) * dtotal
    aV = fc.assemble(a)
    #
    if dircond < 4:
        for bc in bcF:
            bc.apply(aV)
    #
    # solver_V = fc.LUSolver(aV, "mumps")
    solver_V = fc.LUSolver(aV)
    # ------------------------------
    # Initialize Level set function
    # ------------------------------
    phi = fc.Function(V)
    phi.vector()[:] = phi_mat.reshape((Nx)*(Nz))[fc.dof_to_vertex_map(V)]
    cont = 0
    boundaries = fc.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    domains = fc.MeshFunction("size_t", mesh, mesh.topology().dim())
    # -----------------------------
    # Define measures
    # -----------------------------
    dx = Measure('dx')(subdomain_data=domains)
    # -------------------------------
    # Define function Omega1
    # -------------------------------

    class Omega1(fc.SubDomain):
        def inside(self, x, on_boundary):
            return True if phi(x) <= 0 and x[0] >= xMin and x[0] <= xMax and x[1] >= zMin and x[1] <= zMax else False

    # -----------------------------------------------
    # Generating Source and damping
    # -----------------------------------------------
    eta = dmp(parameters)
    plot_mat(parameters, 'Damping', 'Damping function', eta)
    FT = source_config(parameters)
 
    wavesolver = WaveSolver(parameters, eta)
    # -------------------------------
    # Seismograms
    # -------------------------------
    start = time.time()
    d = np.empty((n_shots, Nz, Nx, Nt), np.dtype('float'))
    for stk in range(n_shots):
        d[stk, 0:Nz, 0:Nx, 0:Nt] = wavesolver.measurements(d[stk, 0:Nz, 0:Nx, 0:Nt],
                                                           synth_model,
                                                           FT[stk, 0:Nz, 0:Nx, 0:Nt], 
                                                           add_noise)
    end = time.time()
    # Plot Seismograms
    if wannaPlotFields:
        plotMeasurements(parameters, d)
    print("{:.1f}s to build synthetic seismograms" .format(end - start))

    # auxiliary variables
    phi_mat_old = np.zeros_like(phi_mat)
    vel_field_new = np.zeros_like(vel_field)
    theta1_mat = np.zeros((Nz*Nx))
    theta2_mat = np.zeros_like(theta1_mat)
    weights = []
    MainItEff = 0
    MainIt = 0
    stop_decision = 0
    st_mem_usage = 0.0
    adj_mem_usage = 0.0
    Jevaltotal = np.zeros((MainItMax))
    norm_theta = np.zeros((MainItMax))

    # path to recording phi function
    parameters["path_phi"] = path+'phi/'
    mkDirectory(parameters["path_phi"])
    # path to recording misfit function
    parameters.update({"path_misfit": path+'misfit/'})
    mkDirectory(parameters["path_misfit"])

    # build seismograms
    seismograms = d[:, 0, rec_index, 0:Nt]
    if wannaPlotFields:
        plot_displacement_field(parameters, d[0, 0:Nz, 0:Nx, 0:Nt])
    del(d)

    ###################################################
    # Main Loop
    ###################################################
    while MainIt < MainItMax:
        # ----------------------------------------------
        # Initialize mesh function for boundary domains
        # ----------------------------------------------
        # domains.set_all(0)
        omega1 = Omega1()
        omega1.mark(domains, 1)
        dx = Measure('dx')(subdomain_data=domains)
        J_omega = np.zeros((n_shots))
        u = np.empty((n_shots, Nz, Nx, Nt), np.dtype('float'))
        P = np.empty((n_shots, Nz, Nx, Nt), np.dtype('float'))
        if MainIt > 0:
            vel_field = vel_field_new
        # ------------------------------------
        # Compute STATE. u stands for displacement field
        # ------------------------------------
        start = time.time()
        for stk in range(n_shots):
            u[stk, 0:Nz, 0:Nx, 0:Nt] = stateSolution(
                parameters, eta, u[stk, 0:Nz, 0:Nx, 0:Nt], FT[stk, 0:Nz, 0:Nx, 0:Nt], vel_field)
        end = time.time()
        # ------------------------------------
        # Compute ADJOINT. P stands for the adjoint variable
        # ------------------------------------
        start1 = time.time()
        tr_u = u[0:n_shots, 0, rec_index, 0:Nt]
        misfit = tr_u - seismograms
        for stk in range(n_shots):
            P[stk, 0:Nz, 0:Nx, 0:Nt] = adjointSolution(
                parameters, eta, P[stk, 0:Nz, 0:Nx, 0:Nt], misfit[stk, :, :], vel_field)
        end1 = time.time()
        print('{:.1f}s to compute state and {:.1f}s to compute adjoint with {:d} shots. ' .format(
            end - start, end1-start1, n_shots))

        del(start, end, start1, end1)

        if MainIt > 0:
            vel_field = vel_field_new

        # Plot state/adjoint in 1st-iteration only
        if MainIt == 0 and wannaPlotFields:
            # for s in range(shots):
            mkDirectory(path+'initial_state_%03d/' % (n_shots))
            folder_name = 'initial_state_%03d/' % (n_shots)
            plotstate(parameters, u[0:n_shots, 0:Nz, 0:Nx, 0:Nt], folder_name)
            # plot_displacement_field(parameters, u[1, 0:Nz, 0:Nx, 0:Nt])
            plotadjoint(parameters, P[n_shots-1, 0:Nz, 0:Nx, 0:Nt])
            # Memory usage
        st_mem_usage = (u.size * u.itemsize) / 1_073_741_824  # 1GB
        adj_mem_usage = (P.size * P.itemsize) / 1_073_741_824  # 1GB

        # Plotting reconstructions
        if MainItEff % 10 == 0 or stop_decision == stop_decision_limit-1:
            plottype1(parameters, synth_model, phi_mat, cont)
            plottype2(parameters, synth_model, phi_mat, cont)
            plottype3(parameters, synth_model, phi_mat, MainIt, cont)
            plotcostfunction(parameters, Jevaltotal, MainItEff)
            plotnormtheta(parameters, norm_theta, MainItEff)
            np.save(path+'last_phi_mat.npy', phi_mat)
            gnu_data(phi_mat, 'reconstruction.dat', parameters)
            plot_misfit(parameters, 'misfit', 'Misfit', misfit[:, :, :]) if (
                MainItEff % 50 == 0 and wannaPlotFields) else None

        # -------------------------
        # Compute Cost Function
        # -------------------------
        l2_residual = np.sum(np.power(misfit, 2), axis=1)

        for stk in range(n_shots):
            if MainIt == 0 and add_weight:
                J_omega[stk] += simpson_rule(l2_residual[stk, 0:Nt], gc_t)
                weights.append(1.0e-5)
                # weights.append(J_omega[stk])
            else:
                J_omega[stk] += simpson_rule(l2_residual[stk, 0:Nt], gc_t)

        Jevaltotal[MainItEff] = 0.5*np.sum(J_omega / weights)
        # -------------------------
        # Evaluate shape derivative
        # -------------------------
        start = time.time()
        shapeder = np.reciprocal(weights)[
            0]*derive_sh(parameters, u[0, 0:Nz, 0:Nx, 0:Nt], P[0, 0:Nz, 0:Nx, 0:Nt], eta, V, csi, dx, seism_vel, dtotal)

        for stk in range(1, n_shots):
            shapeder += np.reciprocal(weights)[stk]*derive_sh(
                parameters, u[stk, 0:Nz, 0:Nx, 0:Nt], P[stk, 0:Nz, 0:Nx, 0:Nt], eta, V, csi, dx, seism_vel, dtotal)
        # Build rhs bi-linear system
        shapeder = fc.assemble(shapeder)
        end = time.time()

        print('{}s to compute shape derivative.' .format(end - start))
        del(start, end)
        del(u, P, J_omega)
        with open(path+"cost_function.txt", "a") as file_costfunction:
            file_costfunction.write(
                '{:d} - {:.4e} \n' .format(MainItEff, Jevaltotal[MainItEff]))
        # ====================================
        # ---------- Line search -------------
        # ====================================
        if MainIt > 0 and Jevaltotal[MainItEff] > Jevaltotal[MainItEff - 1] and ls < ls_max:
            ls = ls + 1
            beta = beta * gamma
            phi_mat = phi_mat_old
            # ------------------------------------------------------------
            # Update level set function using the descent direction theta
            # ------------------------------------------------------------
            hj_input = [theta1_mat, theta2_mat,
                        phi_mat, parameters, beta, MainItEff]
            phi_mat = hamiltonjacobi(*hj_input)
            del(hj_input)
            ind = inside_shape(phi_mat)
            ind_c = np.ones_like(phi_mat) - ind
            vel_field_new = seism_vel[0] * ind + seism_vel[1] * ind_c
            phi.vector()[:] = phi_mat.reshape(
                (Nz) * (Nx))[fc.dof_to_vertex_map(V)]
        else:
            print("----------------------------------------------")
            print("Record in: {}" .format(path))
            print("----------------------------------------------")
            print("ITERATION NUMBER (MainItEff)  : {:d}" .format(MainItEff))
            print("ITERATION NUMBER (MainIt)  : {:d}" .format(MainIt))
            print("----------------------------------------------")
            print("Grid Size                : {:d} x {:d}"  .format(Nx, Nz))
            print("State memory usage       : {:.4f} GB" .format(
                st_mem_usage))
            print("Adjoint memory usage     : {:.4f} GB" .format(
                adj_mem_usage))
            print("----------------------------------------------")
            print("Line search  iterations  : {:d}" .format(ls))
            print("Step length beta         : {:.4e}" .format(beta))
            if ls == ls_max:
                beta0 = max(beta0 * gamma2, 0.1 * beta0_init)
            if ls == 0:
                beta0 = min(beta0 / gamma2, 1.0)
            ls = 0
            MainItEff = MainItEff + 1
            beta = beta0  # /(0.999**MainIt)

            # Computing descent direction theta
            if dircond < 4:
                for bc in bcF:
                    bc.apply(shapeder)
            theta = fc.Function(VF)
            solver_V.solve(theta.vector(), -1.0e1*shapeder)

            # ------------------------------------
            # Compute norm theta and grad(phi)
            # ------------------------------------
            theta_vec = theta.vector()[fc.vertex_to_dof_map(VF)]
            theta1_mat = theta_vec[0:len(theta_vec):2].reshape(Nz, Nx)
            theta2_mat = theta_vec[1:len(theta_vec):2].reshape(Nz, Nx)
            norm_theta[MainItEff - 1] = np.sqrt(theta1_mat.reshape(Nz*Nx).dot(theta1_mat.reshape(Nx*Nz))
                                                + theta2_mat.reshape(Nz*Nx).dot(theta2_mat.reshape(Nx*Nz)))
            max_gnp = np.sqrt(fc.assemble(dot(grad(phi), grad(phi))*dtotal))
            print("Norm(grad(phi))          : {:.4e}" .format(max_gnp))
            print(
                "L2-norm of theta         : {:.4e}" .format(norm_theta[MainItEff-1]))
            print("Cost functional          : {:.4e}" .format(
                Jevaltotal[MainItEff-1]))

            # ------------------------------------------------------------
            # Update level set function using the descent direction theta
            # ------------------------------------------------------------
            phi_mat_old = phi_mat
            
            hj_input = [theta1_mat, 
                        theta2_mat,
                        phi_mat, 
                        parameters, 
                        beta, 
                        MainItEff-1]
            
            phi_mat = hamiltonjacobi(*hj_input)

            del(hj_input)
            phi.vector()[:] = phi_mat.reshape(
                (Nz)*(Nx))[fc.dof_to_vertex_map(V)]
            ind = inside_shape(phi_mat)
            ind_c = np.ones_like(phi_mat) - ind
            vel_field_new = seism_vel[0] * ind + seism_vel[1] * ind_c
            
            # ----------------
            # Computing error
            # ----------------
            error_area = np.abs(chi1 - ind)
            relative_error = np.sum(error_area) / np.sum(chi0)
            print('relative error           : {:.3f}%' .format(100*relative_error))

            with open(path+"error.txt", "a") as text_file:
                text_file.write(f'{MainIt} {np.round(relative_error,3):>3}\n')

            # Plot actual phi function
            if MainIt % 50 == 0:
                plot_mat3D(parameters, 'phi_3D', phi_mat, MainIt)
                plot_countour(parameters, 'phi_contour', phi_mat, MainIt)
                phi_ind = '%03d_' % (MainIt)
                np.save(parameters["path_phi"]+phi_ind+'phi.npy', phi_mat)

            # --------------------------------
            # Reinitialize level set function
            # --------------------------------
            if np.mod(MainItEff, 10) == 0:
                phi_mat = reinit(Nz, Nx, phi_mat)

            # ====================================
            # -------- Stopping criterion --------
            # ====================================
            if MainItEff > 5:
                stop0 = stop_coeff * (Jevaltotal[1] - Jevaltotal[2])
                stop1 = Jevaltotal[MainItEff - 2] - Jevaltotal[MainItEff - 1]
                if stop1 < stop0:
                    stop_decision = stop_decision + 1
                if stop_decision == stop_decision_limit:
                    MainIt = MainItMax + 1
                print("stop0                    : {:.4e}"  .format(stop0))
                print("stop1                    : {:.4e}"  .format(stop1))
            print("Stopping step            : {:d} of {:d}"
                  .format(stop_decision, stop_decision_limit))
            print("----------------------------------------------\n")
            cont += 1

        MainIt += 1

    return None


if __name__ == "__main__":
    main(sys.argv[1:])
