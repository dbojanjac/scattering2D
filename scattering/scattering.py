import dolfin as df
import dolfin_utils
import dolfin_utils.meshconvert
from dolfin_utils.meshconvert import meshconvert
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from uuid import uuid4
from subprocess import run, PIPE
import os

if df.MPI.rank(df.mpi_comm_world()) == 0:
    # test version like this because of portability chaos...
    dolfin_version = run(['dolfin-version'],
                                    stdout=PIPE).stdout.decode().strip('\n')
    if dolfin_version != "2017.2.0":
        raise AssertionError("You are using {} ".format(dolfin_version) +
                "FEniCS and code is tested using 2017.2.0 FEniCS version.")


def plane_wave(s, p, k0L):
    #---------------------------------------------------------------------
    # If vectors aren't orthogonal, make them orthogonal by ignoring
    # electric field component in the direction of propagation
    #---------------------------------------------------------------------
    if np.dot(s, p) > 1E-8:
        # Subtract from polarization vector projection in the direction
        # of the propagation
        p = p - (np.dot(s, p) / np.dot(s, s)) * s

    # make unit vectors from s and p
    s = s / np.linalg.norm(s)
    p = p / np.linalg.norm(p)

    pw_r = df.Expression(\
        ('px * cos(k0L * (s_x * x[0] + s_y * x[1]))', \
         'py * cos(k0L * (s_x * x[0] + s_y * x[1]))'),\
        degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

    pw_i = df.Expression(\
        ('px * sin(k0L * (s_x * x[0] + s_y * x[1]))', \
         'py * sin(k0L * (s_x * x[0] + s_y * x[1]))'),\
        degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

    return pw_r, pw_i


def plot_far_field(phi, FF):
    # Put maximum at 0dB
    FF_max = abs(max(np.log10(FF)))
    FF = 10 * (np.log10(FF) + FF_max)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    ax.plot(phi, FF, label = file_name, c = 'b')
    plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
               ncol = 1, mode = "expand", borderaxespad = 0.)
    ax.set_rmax(0);  ax.set_rmin(-10)
    ax.set_rticks([-10, -5, 0, 2])

    plt.plot()
    plt.savefig("ff.png")



class Scattering(ABC):
    '''FEM based solver for electromagnetic wave scattering in 2D on
    heterogeneous isotropic or anisotropic material.

    solve() will solve the sistem and return real and imaginary fields.
    User should pass excitation field of some sort.

    get_far_field() will generate far_field array and return it or save it to
    file if user passed output_file.'''
    def __init__(self, mesh_filename, permittivity_dict, output_dir=None):
        print(permittivity_dict)
        self.permittivity_dict = permittivity_dict

        self.mesh, self.mesh_markers = self.__convert_mesh(mesh_filename)

        self.dx_int = df.Measure('dx', domain=self.mesh,
                subdomain_data=self.mesh_markers)


    def __parse_h5(self, mesh_filename):
        # generate relative mesh_filename, mesh_folder and mesh_name
        rel_mesh_filename = os.path.relpath(mesh_filename)
        mesh_folder = "/".join(mesh_filename.split("/")[:-1])
        mesh_name = mesh_filename.split("/")[-1]

        mesh = df.Mesh()
        hdf = df.HDF5File(mesh.mpi_comm(), rel_mesh_filename, 'r')

        hdf.read(mesh, mesh_folder + "/mesh", False)
        mesh_markers = df.MeshFunction('int', mesh)
        hdf.read(mesh_markers, mesh_folder + "/subdomains")

        return mesh, mesh_markers


    def __call_mesh_convert(self, mesh_filename):
        mesh_id = str(uuid4())
        mesh_xml = "/tmp/" + mesh_id + ".xml"
        meshconvert.convert2xml(mesh_filename, mesh_xml)

        mesh = df.Mesh(mesh_xml)
        without_xml = os.path.splitext(mesh_xml)[0]
        mesh_markers = df.MeshFunction("size_t", mesh, without_xml + "_physical_region.xml");

        return mesh, mesh_markers


    def __convert_mesh(self, mesh_filename):
        if mesh_filename.endswith("h5"):
            mesh, mesh_markers = self.__parse_h5(mesh_filename);
        else:
            if df.MPI.rank(df.mpi_comm_world()) == 0:
                print ("Calling FEniCS meshconvert util")
                mesh, mesh_markers = self._call_mesh_convert(mesh_filename)

        return mesh, mesh_markers


    def solve(self, pw_r, pw_i, k0L):
        #----------------------------------------------------------------------
        # Finite Element function spaces (Nedelec N1curl space),
        # mixed formulation
        #----------------------------------------------------------------------
        NED = df.FiniteElement('N1curl', self.mesh.ufl_cell(), 1)
        V = df.FunctionSpace(self.mesh, 'N1curl', 1)
        W = df.FunctionSpace(self.mesh, NED * NED)

        #----------------------------------------------------------------------
        # Weak formulation (electromagnetic problem is formulated as mixed
        # problem)
        #----------------------------------------------------------------------
        Es_r, Es_i = df.TrialFunctions(W)
        v_r, v_i = df.TestFunctions(W)
        n = df.FacetNormal(self.mesh)

        a_r = df.inner(df.curl(Es_r), df.curl(v_r)) * df.dx \
            - k0L * k0L * df.inner(self.permittivity * Es_r, v_r) * df.dx \
            + k0L * (df.inner(n, Es_i) * df.inner(n, v_r) - df.inner(Es_i, v_r)) * df.ds

        a_i = df.inner(df.curl(Es_i), df.curl(v_i)) * df.dx \
            - k0L * k0L * df.inner(self.permittivity * Es_i, v_i) * df.dx \
            - k0L * (df.inner(n, Es_r) * df.inner(n, v_i) - df.inner(Es_r, v_i)) * df.ds

        L_r = -k0L * k0L * df.inner((self.II - self.permittivity) * pw_r, v_r) * df.dx
        L_i = -k0L * k0L * df.inner((self.II - self.permittivity) * pw_i, v_i) * df.dx

        # Final variational form
        F = a_r + a_i - L_r - L_i

        # Splitting the variational F form into LHS and RHS
        a, L = df.lhs(F), df.rhs(F)

        #----------------------------------------------------------------------
        # Assembling the system (unknown is a vector containing real and
        # imaginary part as two unknown fields, mixed problem)
        #----------------------------------------------------------------------
        # Solution Function
        es = df.Function(W); Es = es.vector()

        # Assemble RHS and LHS
        A = df.assemble(a);    b = df.assemble(L);

        # Solve the system using FEniCS implemented direct solver
        df.solve(A, Es, b)

        #----------------------------------------------------------------------
        # Solution field is equal to sum of incoming and scattered wave
        #----------------------------------------------------------------------
        # Split Solution into real and imaginary part
        Es_r, Es_i = es.split()

        # Interpolate incoming plane wave into N1Curl Function Space
        EI_r = df.interpolate(pw_r, V); EI_i = df.interpolate(pw_i, V)

        # Total Field = Incident Field + Scattered Field
        E_r = Es_r + EI_r;  E_i = Es_i + EI_i

        return E_r, E_i


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


    def get_far_field(self, FF_n, k0L, E_r, E_i, output_file = None):
        step = 1 / float(FF_n - 1)

        phi = np.linspace(step / 2, 2 * np.pi - step / 2, num = FF_n)

        FF_r1 = np.zeros(FF_n); FF_r2 = np.zeros(FF_n)
        FF_i1 = np.zeros(FF_n); FF_i2 = np.zeros(FF_n)
        FF = np.zeros(FF_n)

        # Unit vectors in i and j direction
        e1 = df.as_vector([1, 0]);   e2 = df.as_vector([0, 1])

        # precompute this variables
        rx = np.cos(phi)
        ry = np.sin(phi)

        for n in range (0, FF_n):
            fr = df.Expression('cos(k0L * (rx * x[0] + ry * x[1]))', \
                degree = 1, k0L = k0L, rx = rx[n], ry = ry[n])

            fi = df.Expression('sin(k0L * (rx * x[0] + ry * x[1]))', \
                degree = 1, k0L = k0L, rx = rx[n], ry = ry[n])

            # unit matrix - permittivity_matrx
            A1 = df.as_matrix(((1 - rx[n] * rx[n], rx[n] * ry[n]), (0, 0)))
            A2 = df.as_matrix(((0, 0), (rx[n] * ry[n], 1 - ry[n] * ry[n])))

            #TODO: compile list of subdomains where permittivity is not eq 1
            FF[n] = self._get_ff_component(k0L, A1, A2, E_r, fr, E_i, fi, e1, e2)
        FF = np.sqrt(FF)

        if output_file:
            with open(output_file, "w+") as f:
                for angle, ff in zip(phi, FF):
                    f.write("{} {}\n".format(angle, ff))

        return phi, FF
