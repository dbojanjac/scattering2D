import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def save_field(field, name):
    outfile = fd.File(name)
    outfile.write(field)

def plot_far_field(phi, FF, filename):
    # Put maximum at 0dB
    FF_max = abs(max(np.log10(FF)))
    FF = 10 * (np.log10(FF) + FF_max)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    ax.plot(phi, FF, label = filename, c = 'b')
    plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
               ncol = 1, mode = "expand", borderaxespad = 0.)
    ax.set_rmax(0);  ax.set_rmin(-10)
    ax.set_rticks([-10, -5, 0, 2])

    plt.plot()
    plt.savefig("ff_" + filename + ".png")


class Scattering(ABC):
    def __init__(self, mesh_filename, k0L, output_dir="output"):
        self.mesh = fd.Mesh(mesh_filename)
        self.output_dir = output_dir
        self.k0L = k0L


    def solve(self, incident):
        Ei = incident.interpolate(self.mesh, self.k0L)
        mesh = Ei.ufl_domain()
        V = fd.FunctionSpace(mesh, 'N1curl', 1)

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

        Es = fd.Function(V)
        solver_params = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_view": ""
        }
        fd.solve(a == L, Es, solver_parameters=solver_params)

        E = fd.project(Es + Ei, V)

        return E

    @abstractmethod
    def _get_ff_component(self, k0L, A1, A2, E_r, fr, E_i, fi, e1, e2):
        #----------------------------------------------------------------------
        # Calculating components of far field pattern (real and imaginary part)
        # according to Mischenko: Electromagnetic scattering by Particles and
        # Particle Groups. Cambridge University Press
        #
        # We will split implementation for scalar and tensor values. Scalar
        # implementation can be found in scattering/isotropic_scattering.py
        # and tensor implementation can be found in
        # scattering/anisotropic_scattering.py
        #----------------------------------------------------------------------
        pass


    def get_far_field(self, FF_n, E):
        phi = np.linspace(0, 2 * np.pi, num = FF_n, endpoint = False)
        FF = np.zeros(FF_n)

        # Unit vectors in i and j direction
        # TODO: extract vectors from mesh and make this 3D code
        # e = np.split(np.identity(self.mesh.dim), self.mesh.dim)
        e1 = df.as_vector([1, 0]);   e2 = df.as_vector([0, 1])

        cos_values = np.cos(phi)
        sin_values = np.sin(phi)

        V = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        x = fd.SpatialCoordinate(self.mesh)

        for n in range (0, FF_n):
            rx = cos_values[n]
            ry = cos_values[n]

            r = fd.as_vector([rx, ry])

            f = fd.interpolate(fd.exp(1j * self.k0L * fd.dot(r, x)), V)

            # unit matrix - permittivity_matrx
            A1 = df.Constant([[1 - rx * rx, rx * ry], [0, 0]])
            A2 = df.Constant([[0, 0], [rx * ry, 1 - ry * ry]])

            #TODO: compile list of subdomains where permittivity is not eq 1
            FF[n] = self._get_ff_component(A1, A2, E, f, e1, e2)
        FF = np.sqrt(FF)

        if output_file:
            with open(output_file, "w+") as f:
                for angle, ff in zip(phi, FF):
                    f.write("{} {}\n".format(angle, ff))

        return phi, FF
