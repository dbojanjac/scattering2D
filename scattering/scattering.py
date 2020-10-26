import firedrake as fd
import numpy as np
from abc import ABC, abstractmethod

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


    def solve(self, incident, method="lu"):
        Ei = incident.interpolate(self.mesh, self.k0L)

        V = fd.FunctionSpace(self.mesh, 'N1curl', 1)
        Es = fd.Function(V)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        n = fd.FacetNormal(self.mesh)
        x = fd.SpatialCoordinate(self.mesh)

        epsilon = self.permittivity
        k = self.k0L
        ik = (1j * k)

        a = (fd.inner(fd.curl(u), fd.curl(v)) * fd.dx
            - k**2 * fd.inner(epsilon * u, v) * fd.dx)
        if (self.mesh.topological_dimension() == 3):
                a -= ik * fd.inner((fd.cross(n, fd.cross(n, u))), v) * fd.ds
        else:
                a -= ik * ((fd.dot(u, n) * fd.inner(n, v)) - fd.inner(u, v)) * fd.ds
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

        return fd.project(Es + Ei, V)


    def get_far_field(self, E, FF_n):
        if (self.mesh.topological_dimension() == 3):
            raise NotImplementedError("FF is not straight forward in 3D.")

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
        tdim = self.mesh.topological_dimension()

        r = fd.Constant([1, 0])

        for n in range(FF_n):
            rx = cos_values[n]
            ry = sin_values[n]

            r_vals = np.array([rx, ry])
            r.assign(r_vals)
            A = np.identity(tdim) - np.outer(r_vals, r_vals)

            f = fd.exp(-1j * self.k0L * fd.dot(r, x))

            #TODO: don't integrate if integral is zero
            ffi = fd.assemble(fd.inner((epsilon - self.II) * E * f, v)  * fd.dx)
            # here we have [real_0 + imag_0, ..., real_d + imag_d]
            ff_components = np.sum(ffi.dat.data_ro, axis=0)

            FF[n] = np.linalg.norm(k**2 /(4 * np.pi) * A.dot(ff_components))

        return phi, FF
