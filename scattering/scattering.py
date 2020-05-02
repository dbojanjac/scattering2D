import firedrake as fd
import numpy as np
from abc import ABC, abstractmethod

def save_field(field, name):
    outfile = fd.File(name)
    outfile.write(field)

def plot_far_field(phi, FF, filename):
    import matplotlib.pyplot as plt

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


    def get_far_field(self, E, FF_n):
        phi = np.linspace(0, 2 * np.pi, num = FF_n, endpoint = False)
        FF = np.zeros((FF_n, 2), dtype=np.complex128)

        # Unit vectors in i and j direction
        e1 = fd.as_vector([1, 0])
        e2 = fd.as_vector([0, 1])

        cos_values = np.cos(phi)
        sin_values = np.sin(phi)

        V = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        v = fd.TestFunction(V)
        x = fd.SpatialCoordinate(self.mesh)
        k = self.k0L
        epsilon = self.permittivity

        r = fd.Constant([1, 0])

        for n in range(FF_n):
            rx = cos_values[n]
            ry = sin_values[n]

            r.assign([rx, ry])

            f = fd.exp(1j * self.k0L * fd.dot(r, x))

            #TODO: Constant and assign?
            A = fd.as_matrix([[1 - rx**2, rx * ry],
                             [rx * ry, 1 - ry**2]])

            ff_form = fd.inner(A * (epsilon - self.II) * E * f, v)  * fd.dx
            ff_components = fd.assemble(ff_form)

            # here we have [real_0 + imag_0, ..., real_d + imag_d]
            ff_components = np.sum(ff_components.dat.data_ro, axis=0)
            print ("[DEBUG] ", ff_components)

            # just multiply complex values and get complex values
            FF[n] = k**2 * ff_components / (4 * np.pi)
        # first we sum all components together. After sum we have one complex
        # number and then we compute magnitude. We return real array
        return phi, np.absolute(np.sum(FF, axis = 1))
