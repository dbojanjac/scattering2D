import firedrake as fd
from petsc4py import PETSc
import numpy as np
from abc import ABC, abstractmethod

from .discrete_gradient import build_gradient

class Scattering(ABC):
    def __init__(self, mesh, k0L, output_dir="output"):
        if isinstance(mesh, str):
            self.mesh = fd.Mesh(mesh)
        elif isinstance(mesh, fd.firedrake.mesh.MeshGeometry):
            self.mesh = mesh
        else:
            raise ValueError("Mesh object unknown")
        self.output_dir = output_dir
        self.k0L = k0L

    # smoother for geometric multigrid
    def hybrid_smoother(self, A, b, G):
        GAMMA = 1

        # generate vectors that are needed for algorithm
        with b.dat.vec_ro as b_vec:
            rhs_vec = b_vec.copy()
            r_vec = b_vec.copy()
            g_vec = b_vec.copy()

        # A_phi_mat = G^T * A * G
        A_phi = G.transposeMatMult(A)               # G^T * A
        A_phi = A_phi.matMult(G)                    # G^T * A * G

        g_phi_vec = A_phi.getVecLeft()
        rhs_phi_vec = g_phi_vec.copy()
        iteration = 0
        while True:
            A.mult(g_vec, rhs_vec)                  # rhs = A * g
            r_vec = b_vec - rhs_vec                 # r = b - A * g
            norm = r_vec.norm()

            print (norm)
            iteration += 1

            #TODO: we can remove and have g = G * g_phi
            g_vec.set(0)                            # g     <= 0
            #TODO: add ZERO_INITIAL_GUESS and remove
            g_phi_vec.set(0)                        # g_phi <= 0

            # forward Gauss-Seidel
            G.multTranspose(r_vec, rhs_phi_vec)         # rhs_phi = G^T * r

            # A_phi * g_phi = G^T * r
            sortype_flags = PETSc.Mat.SORType.FORWARD_SWEEP
            A_phi.SOR(rhs_phi_vec, g_phi_vec, sortype=sortype_flags, its=GAMMA)

            # g <= g + G * g_phi
            G.mult(g_phi_vec, rhs_vec)              # rhs = G * g_phi

            g_vec += rhs_vec                        # g  <= g + G *

            # symmetric Gauss-Seidel
            # A * g = r
            sortype_flags = PETSc.Mat.SORType.SYMMETRY_SWEEP
            A.SOR(r_vec, g_vec, sortype=sortype_flags, its=GAMMA)

            # backward Gauss-Seidel
            A.mult(g_vec, rhs_vec)                      # rhs   = A * g
            r_vec -= rhs_vec                            # r = r - A * g
            G.multTranspose(r_vec, rhs_phi_vec)         # rhs = G^T * g_phi

            #TODO: add ZERO_INITIAL_GUESS and remove
            g_phi_vec.set(0)                        # g_phi = 0

            # A_phi * g_phi = rhs
            sortype_flags = PETSc.Mat.SORType.BACKWARD_SWEEP
            A_phi.SOR(rhs_phi_vec, g_phi_vec, sortype=sortype_flags, its=GAMMA)

            G.mult(g_phi_vec, rhs_vec)              # rhs = G * g_phi
            g_vec += rhs_vec                        # g = g + rhs

        return g


    def solve(self, incident, method="lu"):
        Ei = incident.interpolate(self.mesh, self.k0L)
        mesh = Ei.ufl_domain()

        V = fd.FunctionSpace(mesh, 'N1curl', 1)
        Es = fd.Function(V)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        n = fd.FacetNormal(mesh)
        x = fd.SpatialCoordinate(mesh)
        epsilon = self.permittivity
        k = self.k0L
        ik = (1j * k)

        a = (fd.inner(fd.curl(u), fd.curl(v)) * fd.dx
            - k**2 * fd.inner(epsilon * u, v) * fd.dx
            - ik * ((fd.dot(u, n) * fd.inner(n, v)) - fd.inner(u, v)) * fd.ds)
        L = - k**2 * fd.inner((self.II - epsilon) * Ei, v) * fd.dx

        solvers = {
            "lu": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_4": "1"},
        }

        A = fd.assemble(a)
        b = fd.assemble(L)

        solver = fd.LinearSolver(A, solver_parameters=solvers[method])
        solver.solve(Es, b)

        E = fd.project(Es + Ei, V)

        return E


    def get_far_field(self, E, FF_n):
        phi = np.linspace(0, 2 * np.pi, num = FF_n, endpoint = False)
        FF = np.zeros(FF_n)

        cos_values = np.cos(phi)
        sin_values = np.sin(phi)

        V3 = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        V = fd.FunctionSpace(self.mesh, "CG", 1)
        v = fd.TestFunction(V3)
        x = fd.SpatialCoordinate(self.mesh)
        k = self.k0L
        epsilon = self.permittivity

        r = fd.Constant([1, 0])

        for n in range(FF_n):
            rx = cos_values[n]
            ry = sin_values[n]

            r_vals = np.array([rx, ry])
            r.assign(r_vals)
            A = np.identity(2) - np.outer(r_vals, r_vals)

            f = fd.exp(1j * self.k0L * fd.dot(r, x))

            #TODO: don't integrate if integral is zero
            ffi = fd.assemble(fd.inner((epsilon - self.II) * E * f, v)  * fd.dx)
            # here we have [real_0 + imag_0, ..., real_d + imag_d]
            ff_components = np.sum(ffi.dat.data_ro, axis=0)

            FF[n] = np.linalg.norm(k**2 /(4 * np.pi) * A.dot(ff_components))

        return phi, FF
