#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <eigen3/Eigen/Eigen>
#include <omp.h>
//
using namespace std;
namespace py = pybind11;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
//

MatrixXd resizeSlice(py::array_t<double> mat, int index, int nz, int nx)
{
  auto u = mat.mutable_unchecked<3>();
  MatrixXd resize = MatrixXd::Zero(nz + 2, nx + 2);
  for (ssize_t i = 0; i < nz; i++)
  {
#pragma omp simd
    for (ssize_t j = 0; j < nx; j++)
    {
      resize(i + 1, j + 1) = u(i, j, index);
    };
  };

  return resize;
}

py::array_t<double> solve_wv(py::array_t<double> parameters, py::array_t<double> w, py::array_t<double> vel, py::array_t<double> eta, py::array_t<double> S)
{
  auto u = w.mutable_unchecked<3>();
  auto c = vel.unchecked<2>();
  auto damp = eta.unchecked<2>();
  auto source = S.unchecked<3>();
  auto par = parameters.unchecked<1>();
  double hx = par(6);
  double hz = par(7);
  double ht = par(8);
  int nz = u.shape(0);
  int nx = u.shape(1);
  int nt = u.shape(2);
  VectorXd Cxx = VectorXd::Zero(3);
  VectorXd Czz = VectorXd::Zero(3);
  Cxx << 1.0, -2.0, 1.0;
  Czz = Cxx;
  double dzz;
  double dxx;
  double q0;
  double q1;
  double q2;
  double q3;
  double init_cond_u0 = 0.0;
  for (ssize_t i = 0; i < nz; i++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
      u(i, j, 0) = init_cond_u0;
      u(i, j, 1) = u(i, j, 0) + ht * init_cond_u0;
    };
  };

  for (ssize_t t = 1; t < nt - 1; t++)
  {
    auto resized = resizeSlice(w, t, nz, nx);
    for (ssize_t i = 0; i < nz; i++)
    {
#pragma omp simd
      for (ssize_t j = 0; j < nx; j++)
      {
        q0 = c(i, j) * c(i, j) * ht;
        q1 = c(i, j) * c(i, j) * ht * ht;
        q2 = ((c(i, j) * ht) / hx) * ((c(i, j) * ht) / hx);
        q3 = ((c(i, j) * ht) / hz) * ((c(i, j) * ht) / hz);
        dzz = Czz(0) * resized(i, j + 1) + Czz(1) * resized(i + 1, j + 1) + Czz(2) * resized(i + 2, j + 1);
        dxx = Cxx(0) * resized(i + 1, j) + Cxx(1) * resized(i + 1, j + 1) + Cxx(2) * resized(i + 1, j + 2);
        if (i == 0)
        {
          u(0, j, t + 1) = (-1.0 * (u(0, j, t - 1) - 2.0 * u(0, j, t)) + q0 * damp(0, j) * u(0, j, t - 1) + q1 * source(0, j, t + 1) + q2 * dxx + q3 * 2.0 * (u(1, j, t) - u(0, j, t))) * (1.0 / (1.0 + damp(0, j) * q0));
        }
        else
        {
          u(i, j, t + 1) = (-1.0 * (u(i, j, t - 1) - 2.0 * u(i, j, t)) + q0 * damp(i, j) * u(i, j, t - 1) + q1 * source(i, j, t + 1) + q2 * dxx + q3 * dzz) * (1.0 / (1.0 + damp(i, j) * q0));
        };
      };
    };
  };
  return w;
}

MatrixXd pad(py::array_t<double> mat, int index, int nz, int nx)
{
  auto u = mat.mutable_unchecked<3>();
  MatrixXd resize = MatrixXd::Zero(nz + 6, nx + 6);
  for (ssize_t i = 0; i < nz; i++)
  {
#pragma omp simd
    for (ssize_t j = 0; j < nx; j++)
    {
      resize(i + 3, j + 3) = u(i, j, index);
    };
  };

  return resize;
}

py::array_t<double> solve_wv2(py::array_t<double> parameters, py::array_t<double> w, py::array_t<double> vel, py::array_t<double> eta, py::array_t<double> S)
{
  auto u = w.mutable_unchecked<3>();
  auto c = vel.unchecked<2>();
  auto damp = eta.unchecked<2>();
  auto source = S.unchecked<3>();
  auto par = parameters.unchecked<1>();
  double hx = par(6);
  double hz = par(7);
  double ht = par(8);
  int nz = u.shape(0);
  int nx = u.shape(1);
  int nt = u.shape(2);
  /*
  High resolution scheme of 7 points for second derivative
  center coefficients: array([0.01111111, -0.15, 1.5, -2.72222222, 1.5, -0.15,  0.01111111]), 'offsets': array([-3, -2, -1,  0,  1,  2,  3])}, 
  forward coefficients: array([5.21111111, -22.3, 43.95, -52.72222222, 41., -20.1, 5.66111111, -0.7]), 'offsets': array([0, 1, 2, 3, 4, 5, 6, 7])}, 
  backward coefficients: array([-0.7, 5.66111111, -20.1, 41., -52.72222222, 43.95, -22.3,  5.21111111]), 'offsets': array([-7, -6, -5, -4, -3, -2, -1,  0])}}
  
  for the term Gamma_n
  High resolution scheme of 7 points for first derivative
  center coefficients: array([-1.66666667e-02, 1.50000000e-01, -7.50000000e-01, 0.0, 7.50000000e-01, -1.50000000e-01, 1.66666667e-02]), 'offsets': array([-3, -2, -1, 0, 1, 2, 3])}, 
  forward coefficients: array([-2.45, 6., -7.5, 6.66666667, -3.75, 1.2, -0.16666667]), 'offsets': array([0, 1, 2, 3, 4, 5, 6])}, 
  backward coefficients: array([0.16666667, -1.2, 3.75, -6.66666667, 7.5, -6., 2.45]), 'offsets': array([-6, -5, -4, -3, -2, -1,  0])}}
 
  for the term gamma_n2
  center coefficient: array([0.08333333, -0.66666667, 0., 0.66666667, -0.08333333]), 'offsets': array([-2,-1, 0, 1, 2])}, 
  forward coefficients: array([-2.08333333, 4., -3., 1.33333333, -0.25]), 'offsets': array([0, 1, 2, 3, 4])}, 
  backward coefficients: array([0.25, -1.33333333, 3., -4., 2.08333333]), 'offsets': array([-4, -3,-2,-1,0])}}
    */
  VectorXd Cxx = VectorXd::Zero(7);
  VectorXd Czz = VectorXd::Zero(7);
  VectorXd Cz = VectorXd::Zero(7);
  VectorXd Bz = VectorXd::Zero(5);
  Cxx << 0.01111111, -0.15, 1.5, -2.72222222, 1.5, -0.15, 0.01111111;
  Czz = Cxx;
  Cz << -1.66666667e-02, 1.50000000e-01, -7.50000000e-01, 0.0, 7.50000000e-01, -1.50000000e-01, 1.66666667e-02;
  double dzz;
  double dzz0;
  double dxx;
  double gamma_n;
  double q0;
  double q1;
  double q2;
  double q3;
  double init_cond_u0 = 0.0;
  for (ssize_t i = 0; i < nz; i++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
      u(i, j, 0) = init_cond_u0;
      u(i, j, 1) = u(i, j, 0) + ht * init_cond_u0;
    };
  };

  for (ssize_t t = 1; t < nt - 1; t++)
  {
    auto resized = pad(w, t, nz, nx);
    for (ssize_t i = 0; i < nz; i++)
    {
#pragma omp simd
      for (ssize_t j = 0; j < nx; j++)
      {
        q0 = c(i, j) * c(i, j) * ht;
        q1 = c(i, j) * c(i, j) * ht * ht;
        q2 = ((c(i, j) * ht) / hx) * ((c(i, j) * ht) / hx);
        q3 = ((c(i, j) * ht) / hz) * ((c(i, j) * ht) / hz);
        dzz = Czz(0) * resized(i, j + 3) + Czz(1) * resized(i + 1, j + 3) + Czz(2) * resized(i + 2, j + 3) + Czz(3) * resized(i + 3, j + 3) + Czz(4) * resized(i + 4, j + 3) + Czz(5) * resized(i + 5, j + 3) + Czz(6) * resized(i + 6, j + 3);

        dxx = Cxx(0) * resized(i + 3, j) + Cxx(1) * resized(i + 3, j + 1) + Cxx(2) * resized(i + 3, j + 2) + Cxx(3) * resized(i + 3, j + 3) + Cxx(4) * resized(i + 3, j + 4) + Cxx(5) * resized(i + 3, j + 5) + Cxx(6) * resized(i + 3, j + 6);

        if (i == 0)
        {
          gamma_n = (-1 / Cz(0)) * (Cz(1) * resized(i + 1, j + 3) + Cz(2) * resized(i + 2, j + 3) + Cz(4) * resized(i + 4, j + 3) + Cz(5) * resized(i + 5, j + 3) + Cz(6) * resized(i + 6, j + 3));

          dzz0 = Czz(0) * gamma_n + Czz(1) * resized(i + 1, j + 3) + Czz(2) * resized(i + 2, j + 3) + Czz(3) * resized(i + 3, j + 3) + Czz(4) * resized(i + 4, j + 3) + Czz(5) * resized(i + 5, j + 3) + Czz(6) * resized(i + 6, j + 3);

          u(0, j, t + 1) = (-1.0 * (u(0, j, t - 1) - 2.0 * u(0, j, t)) + q0 * damp(0, j) * u(0, j, t - 1) + q1 * source(0, j, t) + q2 * dxx + q3 * dzz0) * (1.0 / (1.0 + damp(0, j) * q0));
        }
        else
        {
          u(i, j, t + 1) = (-1.0 * (u(i, j, t - 1) - 2.0 * u(i, j, t)) + q0 * damp(i, j) * u(i, j, t - 1) + q1 * source(i, j, t) + q2 * dxx + q3 * dzz) * (1.0 / (1.0 + damp(i, j) * q0));
        };
      };
    };
  };
  return w;
}

py::array_t<double> reverse_time(py::array_t<double> p, py::array_t<double> p_tilde)
{
  auto P = p.mutable_unchecked<3>();
  auto ptilde = p_tilde.mutable_unchecked<3>();
  int nz = ptilde.shape(0);
  int nx = ptilde.shape(1);
  int nt = ptilde.shape(2);

  for (ssize_t t = 0; t < nt; t++)
  {
    for (ssize_t i = 0; i < nz; i++)
    {
      for (ssize_t j = 0; j < nx; j++)
      {
        P(i, j, t) = ptilde(i, j, (nt - 1) - t);
      };
    };
  };
  return p;
}

// The level set method for 2d - domain
py::array_t<double> hj(py::array_t<double> v1, py::array_t<double> v2, py::array_t<double> phiraw, py::array_t<double> parameters)
{
  auto phi = phiraw.mutable_unchecked<2>();
  auto V1 = v1.unchecked<2>();
  auto V2 = v2.unchecked<2>();
  auto par = parameters.unchecked<1>();
  double beta = par(0);
  double itermax = par(1);
  int nz = phi.shape(0);
  int nx = phi.shape(1);
  double delta_z = par(2);
  double delta_x = par(3);
  double h = min(delta_z, delta_x);
  double maxv;
  double dt;
  MatrixXd maxv_mat = MatrixXd::Zero(nz, nx);
  MatrixXd g = MatrixXd::Zero(nz, nx);
  MatrixXd Dxp = MatrixXd::Zero(nz, nx);
  MatrixXd Dyp = MatrixXd::Zero(nz, nx);
  // -------- x --------
  MatrixXd Dxm = MatrixXd::Zero(nz, nx);
  MatrixXd Dym = MatrixXd::Zero(nz, nx);
  for (ssize_t k = 0; k < itermax; k++)
  {
    // Compute first-order terms in the HJ equation
    Dxp.setZero();
    Dyp.setZero();
    Dxm.setZero();
    Dym.setZero();
    /* Be careful here,
       the x-axis is in column
       the y-axis is in rows
     */
    for (ssize_t i = 0; i < nz; i++)
    {
      for (ssize_t j = 0; j < nx - 1; j++)
      {
        Dxp(i, j) = (phi(i, j + 1) - phi(i, j)) / delta_x;
      }
    }

    for (ssize_t i = 0; i < nz; i++)
    {
      for (ssize_t j = 1; j < nx; j++)
      {
        Dxm(i, j) = (phi(i, j) - phi(i, j - 1)) / delta_x;
      }
    }

    for (ssize_t i = 0; i < nz - 1; i++)
    {
      for (ssize_t j = 0; j < nx; j++)
      {
        Dyp(i, j) = (phi(i + 1, j) - phi(i, j)) / delta_z;
      }
    }

    for (ssize_t i = 1; i < nz - 1; i++)
    {
      for (ssize_t j = 0; j < nx; j++)
      {
        Dym(i, j) = (phi(i, j) - phi(i - 1, j)) / delta_z;
      }
    }

    for (ssize_t i = 0; i < nz; i++)
    {
      Dxp(i, nx - 1) = Dxp(i, nx - 2);
      Dxm(i, 0) = Dxm(i, 1);
    }

    for (ssize_t j = 0; j < nx; j++)
    {
      Dyp(nz - 1, j) = Dxp(nz - 2, j);
      Dym(0, j) = Dym(1, j);
    }

#pragma omp parallel for schedule(static, 16)
    for (ssize_t i = 0; i < nz; i++)
    {
#pragma omp simd
      for (ssize_t j = 0; j < nx; j++)
      {
        g(i, j) = 0.5 * (V1(i, j) * (Dxp(i, j) + Dxm(i, j)) + V2(i, j) * (Dyp(i, j) + Dym(i, j))) - 0.5 * abs(V1(i, j)) * (Dxp(i, j) - Dxm(i, j)) - 0.5 * abs(V2(i, j)) * (Dyp(i, j) - Dym(i, j));
        maxv_mat(i, j) = abs(V1(i, j)) + abs(V2(i, j));
      }
    }

    maxv = maxv_mat.maxCoeff();
    dt = beta * h / (2 * sqrt(2) * maxv);

    for (ssize_t i = 0; i < nz; i++)
    {
      for (ssize_t j = 0; j < nx; j++)
      {
        phi(i, j) = phi(i, j) - dt * g(i, j);
      }
    }
  }
  return phiraw;
}

// The absorbing damping layer for the acoustic wave equation
py::array_t<double> damping_function(py::array_t<double> eta_raw, py::array_t<double> parameters)
{
  auto par = parameters.unchecked<1>();
  double xMin = par(0);
  double xMax = par(1);
  double zMin = par(2);
  double zMax = par(3);
  double hx = par(6);
  double hz = par(7);
  double dmp_xmin = par(9);
  double dmp_xmax = par(10);
  double dmp_zmax = par(11);
  //
  int nz = (zMax - zMin) / hz;
  int nx = (xMax - xMin) / hx;
  VectorXd grid_x, grid_z;
  grid_x.setLinSpaced(nx, xMin, xMax);
  grid_z.setLinSpaced(nz, zMin, zMax);
  auto dmp_func = [](auto x, auto lim) { return 1e4 * pow(abs(x - lim), 2); };
  //
  // MatrixXd eta = MatrixXd::Zero(nz, nx);
  auto eta = eta_raw.mutable_unchecked<2>();
  double a1 = (zMax - dmp_zmax) / (xMin - dmp_xmin);
  double a2 = (zMax - dmp_zmax) / (xMax - dmp_xmax);
  double b1 = (dmp_zmax * xMin - zMax * dmp_xmin) / (xMin - dmp_xmin);
  double b2 = (dmp_zmax * xMax - zMax * dmp_xmax) / (xMax - dmp_xmax);
  //
#pragma omp parallel for collapse(2) schedule(static)
  for (ssize_t i = 0; i < nz; i++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
      if ((grid_x(j) < dmp_xmin && grid_z(i) < dmp_zmax) || (grid_z(i) >= dmp_zmax && grid_z(i) <= a1 * grid_x(j) + b1))
      {
        eta(i, j) = dmp_func(grid_x(j), dmp_xmin);
      }
      else if ((grid_x(j) > dmp_xmax && grid_z(i) < dmp_zmax) || (grid_z(i) >= dmp_zmax && grid_z(i) <= a2 * grid_x(j) + b2))
      {
        eta(i, j) = dmp_func(grid_x(j), dmp_xmax);
      }
      else if (grid_z(i) >= dmp_zmax && ((grid_x(j) >= 0 && grid_x(j) <= 1.0 / a1 * (grid_z(i) - b1)) || grid_x(j) <= 1.0 / a2 * (grid_z(i) - b2)))
      {
        eta(i, j) = dmp_func(grid_z(i), dmp_zmax);
      }
    }
  }
  return eta_raw;
}

const auto chunk = 16;
const auto num_collapse = 2;

py::array_t<double> dt_u(py::array_t<double> u, py::array_t<double> ut, float deltat)
{
  auto U = u.mutable_unchecked<3>();
  auto Ut = ut.mutable_unchecked<3>();
  int nz = u.shape(0);
  int nx = u.shape(1);
  int nt = u.shape(2);
#pragma omp parallel for collapse(num_collapse) schedule(static, chunk)
  for (ssize_t i = 0; i < nz; i++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
#pragma omp simd
      for (ssize_t t = 0; t < nt - 1; t++)
      {
        Ut(i, j, t + 1) = (1.0 / deltat) * (U(i, j, t + 1) - U(i, j, t));
      }
      Ut(i, j, 0) = 0.0;
    }
  }
  return ut;
}

py::array_t<double> dt_p(py::array_t<double> p, py::array_t<double> pt, float deltat)
{
  auto P = p.mutable_unchecked<3>();
  auto Pt = pt.mutable_unchecked<3>();
  int nz = p.shape(0);
  int nx = p.shape(1);
  int nt = p.shape(2);
#pragma omp parallel for collapse(num_collapse) schedule(static, chunk)
  for (ssize_t i = 0; i < nz; i++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
#pragma omp simd
      for (ssize_t t = 0; t < nt - 1; t++)
      {
        Pt(i, j, t + 1) = (1.0 / deltat) * (P(i, j, t + 1) - P(i, j, t));
      }
      Pt(i, j, 0) = Pt(i, j, 1);
    }
  }
  return pt;
}

py::array_t<double> dx_cpp(py::array_t<double> u, py::array_t<double> u_x, float deltax)
{
  auto U = u.mutable_unchecked<3>();
  auto Ux = u_x.mutable_unchecked<3>();
  int nz = u.shape(0);
  int nx = u.shape(1);
  int nt = u.shape(2);
#pragma omp parallel for collapse(num_collapse) schedule(static, chunk)
  for (ssize_t t = 0; t < nt; t++)
  {
    for (ssize_t i = 0; i < nz; i++)
    {
#pragma omp simd
      for (ssize_t j = 1; j < nx - 1; j++)
      {
        Ux(i, j, t) = 0.5 * (1.0 / deltax) * (U(i, j + 1, t) - U(i, j - 1, t));
      }
      Ux(i, 0, t) = Ux(i, 1, t);
      Ux(i, nx - 1, t) = Ux(i, nx - 2, t);
    }
  }
  return u_x;
}

py::array_t<double> dz_cpp(py::array_t<double> u, py::array_t<double> u_z, float deltaz)
{
  auto U = u.mutable_unchecked<3>();
  auto Uz = u_z.mutable_unchecked<3>();
  int nz = u.shape(0);
  int nx = u.shape(1);
  int nt = u.shape(2);
#pragma omp parallel for collapse(num_collapse) schedule(static, chunk)
  for (ssize_t t = 0; t < nt; t++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
#pragma omp simd
      for (ssize_t i = 1; i < nz - 1; i++)
      {
        Uz(i, j, t) = 0.5 * (1.0 / deltaz) * (U(i + 1, j, t) - U(i - 1, j, t));
      }
      Uz(0, j, t) = Uz(1, j, t);
      Uz(nz - 1, j, t) = Uz(nz - 2, j, t);
    }
  }
  return u_z;
}

py::array_t<double> int_0T(py::array_t<double> k, py::array_t<double> du_, py::array_t<double> dp_, py::array_t<double> coeffs)
{
  auto K = k.mutable_unchecked<1>();
  auto du = du_.mutable_unchecked<3>();
  auto dp = dp_.mutable_unchecked<3>();
  auto coeff = coeffs.mutable_unchecked<1>();
  int nz = du_.shape(0);
  int nx = du_.shape(1);
  int nt = du_.shape(2);
#pragma omp parallel for collapse(num_collapse) schedule(static, chunk)
  for (ssize_t i = 0; i < nz; i++)
  {
    for (ssize_t j = 0; j < nx; j++)
    {
#pragma omp simd
      for (ssize_t t = 0; t < nt; t++)
      {
        K(i * nx + j) += coeff(t) * du(i, j, t) * dp(i, j, t);
      }
    }
  }
  return k;
}

// ---------------- X -------------------
PYBIND11_MODULE(c_functions, m)
{
  m.doc() = R"pbdoc(
        Pybind11 wave solver plugin
        -----------------------
     
        .. currentmodule:: c_functions
     
        .. autosummary::
           :toctree: _generate
         
           solver
    )pbdoc";

  m.def("solve_wv", &solve_wv, R"pbdoc(

        Gives the solution through finite difference of acoustic wave PDE rho*u_tt - eta*u_t + div(grad(u) = f. Given the velocity field c = 1/rho**2, some damping function eta and the source f. 
    )pbdoc");

  m.def("solve_wv2", &solve_wv2, R"pbdoc(

        Gives the solution through 7-points finite difference scheme of acoustic wave PDE rho*u_tt - eta*u_t + div(grad(u) = f. Given the velocity field c = 1/rho**2, some damping function eta and the source f. 
    )pbdoc");

  m.def("reverse_time", &reverse_time, R"pbdoc(

        Invert order of the third dimension of some tree-dimensional Matrix;
    )pbdoc");

  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
     
        Some other explanation about the subtract function.
    )pbdoc");

  m.def("hj", &hj, R"pbdoc(
          
        Compute level set function;
    )pbdoc");

  m.def("damping_function", &damping_function, R"pbdoc(
          
        Compute the absorbing sponge layer for the acoustic wave equation;
    )pbdoc");

  m.def("dt_u", &dt_u, R"pbdoc(Calculate state derivative with respect to time variable t. This function uses forward first order finite difference formula)pbdoc");

  m.def("dt_p", &dt_p, R"pbdoc(Calculate adjoint derivative with respect to time variable t. This function uses forward first order finite difference formula)pbdoc");

  m.def("dx_cpp", &dx_cpp, R"pbdoc(Calculate state derivative with respect to space variable x in x-axis direction. This function uses a centered first order finite difference formula to calculate the discrete derivative.)pbdoc");

  m.def("dz_cpp", &dz_cpp, R"pbdoc(Calculate state derivative with respect to space variable z in z-axis direction. This function uses a centered first order finite difference formula to calculate the discrete derivative.)pbdoc");

  m.def("int_0T", &int_0T, R"pbdoc(Calculate int_0^T <du_ * d_p> terms of shape derivative. Uses simpson rule to integrate over time.)pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
