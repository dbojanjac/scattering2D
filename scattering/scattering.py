import firedrake as fd
import numpy as np
from abc import ABC, abstractmethod

class ExcitationBase(ABC):
    """Abstract Excitation object"""

    @abstractmethod
    def interpolate(self, mesh, *args):
        """Interpolate Excitation on the mesh"""
        pass


class Scattering(ABC):
    """
    Base class for all Scattering problems.
    Scattering object is wrapper above problem description and domain where problem
    is defined.
    """

    def __init__(self, mesh, k0L):
        """
        Initialize Scattering object

        Parameters:
        -----------
        mesh: either path to the file with extension of some supported mesh format[1] or
              already created firedrake mesh. One should use firedrake mesh in order to
              do a errornorm between two solutions on the same mesh.
        k0L:  wave number

        [1] - https://www.firedrakeproject.org/firedrake.html#firedrake.mesh.Mesh
        """
        if isinstance(mesh, str):
            self.mesh = fd.Mesh(mesh)
        elif isinstance(mesh, fd.firedrake.mesh.MeshGeometry):
            self.mesh = mesh
        else:
            raise ValueError("Mesh object unknown")
        self.k0L = k0L


    def solve(self, incident, method="lu"):
        """
        Solve Maxwell Scattering problem using incident excitation

        We are solving:

        curl(curl(E)) - k^2 epsilon_r * E = 0
        E = E_i + E_s where E_i is incident wave

        As a boundary conditions we are using first order Silver-Muller BC.

        Function will compute E_s and return E = E_i + E_s.

        Parameters:
        -----------
        incident: object that should inherit from ExcitationBase
        method:   method for solving Ax=b system. Currently we default to MUMPS.
        """
        if isinstance(incident, ExcitationBase):
            Ei = incident.interpolate(self.mesh, self.k0L)
        else:
            raise ValueError("incident object unknown")

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
        if self.mesh.topological_dimension() == 3:
                a -= ik * fd.inner((fd.cross(n, fd.cross(n, u))), v) * fd.ds
        else:
                # for 2D case we have to write a \times b \times c as:
                # (a \cdot c) \cdot b - (a \cdot b) \cdot c
                a -= ik * ((fd.dot(n, u) * fd.inner(n, v)) - fd.inner(u, v)) * fd.ds
        L = - k**2 * fd.inner((self.II - epsilon) * Ei, v) * fd.dx

        solvers = {
            "lu": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_4": "1"},
        }

        #TODO: LinearVariationalSolver
        A = fd.assemble(a)
        b = fd.assemble(L)

        solver = fd.LinearSolver(A, solver_parameters=solvers[method])
        solver.solve(Es, b)

        return fd.project(Es + Ei, V)


    def get_far_field(self, E, ff_num_samples, theta=np.pi/2):
        """Compute Far field for the electric field

        Compute far field components using Mischenkos: Electromagnetic scattering by Particles
        and Particle Groups. Cambridge University Press.

        Parameters:
        -----------
        E:              electromagnetic field for which we are computing far field.
        ff_num_samples: number of far field components. In other word, how much values will
                        we sample along the circle. Points on the circle are equidistant.
        theta:          angle with respect to polar axis. pi/2 generates xOy plane.

        Returns:
        --------
        phi:     np.array of ff_num_samples angles where we computed far field.
        ff_data: np.array of ff_num_samples which represent far field at each angle phi.
        """
        phi = np.linspace(0, 2 * np.pi, num = ff_num_samples, endpoint = False)
        ff_data = np.zeros(ff_num_samples)

        V3 = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        V = fd.FunctionSpace(self.mesh, "CG", 1)
        v = fd.TestFunction(V3)
        x = fd.SpatialCoordinate(self.mesh)
        k = self.k0L
        epsilon = self.permittivity
        tdim = self.mesh.topological_dimension()

        if tdim == 2:
            r = fd.Constant([1, 0])
        else:
            r = fd.Constant([1, 0, 0])

        for n in range(ff_num_samples):
            if tdim == 2:
                r_vals = [np.cos(phi[n]),
                          np.sin(phi[n])]
            else:
                r_vals = [np.sin(theta) * np.cos(phi[n]),
                          np.sin(theta) * np.sin(phi[n]),
                          np.cos(theta)]

            r.assign(r_vals)
            A = np.identity(tdim) - np.outer(r_vals, r_vals)

            f = fd.exp(-1j * self.k0L * fd.dot(r, x))

            #TODO: don't integrate if integral is zero
            ffi = fd.assemble(fd.inner((epsilon - self.II) * E * f, v)  * fd.dx)
            # here we have [real_0 + imag_0, ..., real_d + imag_d]
            ff_components = np.sum(ffi.vector().gather().reshape(-1, tdim), axis=0)

            # ff_components = [sum_along_0, sum_along_1, ..., sum_along_d]
            ff_data[n] = np.linalg.norm(k**2 /(4 * np.pi) * A.dot(ff_components))

        return phi, ff_data
