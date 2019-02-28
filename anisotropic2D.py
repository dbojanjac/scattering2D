# FEM based solver for electromagnetic wave scattering in 2D on anisotropic material

    # Incoming plane wave parameters such as frequency k0L,  polarization p
    # and direction s (E = p exp(i * k0L * s.x)) are hardcoded in main part as:
    #   s = [1, 2];    p = [-2, 1];  k0L = 3.141592653589793
    # while permittivity for anisotropic medium is stored in perrmitivity_file as
    # 2x2 matrix

# Function call: python3 anisotropic2D.py input_folder mesh_name permittivity_file output_folder FF_n
# ie. python3 anisotropic2D.py mesh anisotropic effective results 72

# input = domain mesh with subdomain markers in .h5 format
# output = real and imaginary part of total electric field and far field pattern


# Using FEniCS 2017.2.0
import dolfin as df
import numpy as np
import sys


def mesh_anisotropic_2D(input_folder, mesh_name, \
    permittivity_file, air_permittivity):
    """Read mesh and subdomains, from .h5 mesh file, for
        homogeneous anisotropic domain"""

    # Input Variables:
        # input_folder: mesh contaning folder
        # mesh_name: name of the mesh file (in .h5 format), mesh subdomains are
                   # stored in subdomains part of input_folder/mesh_name.h5 file
                   # coefficients in elements:
                        #   1 represents anisotropic material
                        #   3 outside material (air_permittivity)

    # Output Variables:
        # mesh: FEniCS mesh variable
        # markers: MeshFunction describing mesh subdomains according to:
                        #   1 represents outside material (air_permittivity)
                        #   2 represents material matrix (matrix_permittivity)
                        #   3 represents inner material (patch_permittivity)

        # permittivity: zeroth order polynomial, permittivity function

    # Read mesh and subdomain markers from input_folder/mesh_name
    #---------------------------------------------------------------------------
    input_folder = input_folder + '/'

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), input_folder + mesh_name + '.h5', 'r')
    hdf.read(mesh, input_folder + "mesh", False)
    markers = df.MeshFunction('int', mesh)
    hdf.read(markers, input_folder + "subdomains")

    #---------------------------------------------------------------------------
    # Permittivity coefficient for previously defined subdomains
    #---------------------------------------------------------------------------

    # Read permittivity tensor from permittivity_file
    ifile = open(permittivity_file, 'r')

    eps_input = [None] * 4; k = 0
    for line in ifile:
        coeff = line.split()
        for m in range(0, 2):
            eps_input[m + 2 * k] = float(coeff[m])
        k += 1

    # Domain definition coefficients
    class Coeff(df.Expression):

        def __init__(self, mesh, **kwargs):
            self.markers = markers

        def eval_cell(self, values, x, cell):

            if markers[cell.index] == 3:
                values[0] = air_permittivity
                values[1] = 0
                values[2] = 0
                values[3] = air_permittivity

            if markers[cell.index] == 1:
                values[0] = eps_input[0]
                values[1] = eps_input[1]
                values[2] = eps_input[2]
                values[3] = eps_input[3]

        def value_shape(self):
            return (2,2)

    # Interpolation to zeroth order polynomial
    permittivity = Coeff(mesh, degree = 0)

    return mesh, markers, permittivity
#-------------------------------------------------------------------------------


def plane_wave_2D(s, p, k0L):
    """Plane Wave excitation E = p exp(i * k0L * s.x)"""

    # Input Variables:
        # s: plane wave direction of propagation (unit vector)
        # p: plane wave polarization (unit vector)
        # k0L: wave number (positive real number)

    # Output Variables:
        # pw_r: real part of incoming plane wave
        # pw_i: imaginary part of incoming plane wave

    #---------------------------------------------------------------------------
    # Check if polarization is orthogonal to direction of propagation
    #---------------------------------------------------------------------------
    if (s[0] * p[0] + s[1] * p[1]) <= 1E-8:

        # make unit vectors from s and p
        s_norm = np.sqrt(s[0] ** 2 + s[1] ** 2)
        s[0], s[1] = s[0] / s_norm, s[1] / s_norm

        p_norm = np.sqrt(p[0] ** 2 + p[1] ** 2)
        p[0], p[1] = p[0] / p_norm, p[1] / p_norm

        pw_r = df.Expression(\
            ('px * cos(k0L * (s_x * x[0] + s_y * x[1]))', \
            'py * cos(k0L * (s_x * x[0] + s_y * x[1]))'), \
            degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

        pw_i = df.Expression(\
            ('px * sin(k0L * (s_x * x[0] + s_y * x[1]))', \
            'py * sin(k0L * (s_x * x[0] + s_y * x[1]))'), \
            degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

    #---------------------------------------------------------------------------
    # Make it orthogonal by ignoring electric field component in the
    # direction of propagation
    #---------------------------------------------------------------------------
    else:

        # Subtract from polarization vector projection in the direction of the propagation
        p[0] = p[0] - ((s[0] * p[0] + s[1] * p[1]) \
            / (s[0] * s[0] + s[1] * s[1])) * s[0]

        p[1] = p[1] - ((s[0] * p[0] + s[1] * p[1]) \
            / (s[0] * s[0] + s[1] * s[1])) * s[1]

        # make unit vectors from s and p
        s_norm = np.sqrt(s[0] ** 2 + s[1] ** 2)
        s_x, s_y = s[0] / s_norm, s[1] / s_norm

        p_norm = np.sqrt(p[0] ** 2 + p[1] ** 2)
        px, py = p[0] / p_norm, p[1] / p_norm

        pw_r = df.Expression(\
            ('px * cos(k0L * (s_x * x[0] + s_y * x[1]))', \
            'py * cos(k0L * (s_x * x[0] + s_y * x[1]))'), \
            degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

        pw_i = df.Expression(\
            ('px * sin(k0L * (s_x * x[0] + s_y * x[1]))', \
            'py * sin(k0L * (s_x * x[0] + s_y * x[1]))'), \
            degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

    return pw_r, pw_i
#-------------------------------------------------------------------------------


def solver_anisotropic_2D(mesh, permittivity, pw_r, pw_i, k0L):
    """Electromagnetic scattering solver based on FEM"""

    # Input Variables:
        # mesh: mesh keeping variable
        # permittivity: permittivity variable (anisotropic 2x2 matrix function)
        # pw_r: real part of incoming plane wave
        # pw_i: imaginary part of incoming plane wave
        # k0L: name of .h5 file in which mesh is stored

    # Output Variables:
        # Er: real part of total field
        # Ei: imaginary part of total field

    # Nedelec Elements Function space
    NED = df.FiniteElement('N1curl', mesh.ufl_cell(), 1)
    V = df.FunctionSpace(mesh, 'N1curl', 1)
    W = df.FunctionSpace(mesh, NED * NED)

    # Unit matrix
    II = df.as_matrix(((1, 0), (0,1)))

    #---------------------------------------------------------------------------
    # Weak formulation (electromagnetic problem is formulated as mixed problem)
    #---------------------------------------------------------------------------
    Es_r, Es_i = df.TrialFunctions(W)
    v_r, v_i = df.TestFunctions(W)
    n = df.FacetNormal(mesh)

    a_r = df.inner(df.curl(Es_r), df.curl(v_r)) * df.dx \
        - k0L * k0L * df.inner(permittivity * Es_r, v_r) * df.dx \
        + k0L * (df.inner(n, Es_i) * df.inner(n, v_r) - df.inner(Es_i, v_r)) * df.ds

    a_i = df.inner(df.curl(Es_i), df.curl(v_i)) * df.dx \
        - k0L * k0L * df.inner(permittivity * Es_i, v_i) * df.dx \
        - k0L * (df.inner(n, Es_r) * df.inner(n, v_i) - df.inner(Es_r, v_i)) * df.ds

    L_r = - k0L * k0L * df.inner((II - permittivity) * pw_r, v_r) * df.dx
    L_i = - k0L * k0L * df.inner((II - permittivity) * pw_i, v_i) * df.dx

    # Final variational form
    F = a_r + a_i - L_r - L_i

    # Splitting the variational form into LHS and RHS
    a, L = df.lhs(F), df.rhs(F)

    #---------------------------------------------------------------------------
    # Assembling the system (unknown is a vector containing real and imaginary
    # part as two unknown fields, mixed problem)
    #---------------------------------------------------------------------------
    # Solution Function
    es = df.Function(W); Es = es.vector()

    # Assemble RHS, LHS
    A = df.assemble(a);    b = df.assemble(L);

    # Solve the system using FEniCS implemented direct solver
    df.solve(A, Es, b)

    #---------------------------------------------------------------------------
    # Solution field is equal to sum of incoming and scattered wave
    #---------------------------------------------------------------------------
    # Split Solution into real and imaginary part
    Es_r, Es_i = es.split()

    # Interpolate incoming plane wave into N1Curl Function Space
    EI_r = df.interpolate(pw_r, V); EI_i = df.interpolate(pw_i, V)

    # Total Field = Incident Field + Scattered Field
    Er = Es_r + EI_r;  Ei = Es_i + EI_i

    return Er, Ei
#-------------------------------------------------------------------------------


def ff_anisotropic_2D(mesh_name, output_folder, permittivity, k0L, E_r, E_i, FF_n):
    """Far Field calculator"""

    # Input Variables:
        # mesh_name: mesh keeping variable
        # output_folder:
        # permittivity: permittivity function
        # k0L: dimensionless parameter describing wave vector length
        # E_r: real part of scattered electric field
        # E_i: imaginary part of scattered electric field
        # FF_n: number of far field pattern sample points

    # Output Variables:
        # phi: angle list from [0, 2pi]
        # FF: far field value

    # Used Variables:
        # step: step length in phi discretization
        # FF_r*: * component of real part of far field pattern
        # FF_i*: * component of imaginary part of far field pattern

    step = 1 / float(FF_n - 1)

    phi = np.linspace(step / 2, 2 * 3.1415 - step / 2, num = FF_n)

    FF_r1 = [0] * FF_n; FF_r2 = [0] * FF_n
    FF_i1 = [0] * FF_n; FF_i2 = [0] * FF_n
    FF = [0] * FF_n;

    # Unit 2x2 matrix
    II = df.as_matrix(((1, 0), (0,1)))

    # Unit vectors in i and j direction
    e1 = df.as_vector([1, 0]);   e2 = df.as_vector([0, 1])

    for n in range (0, FF_n):

        # Sampled unit vector on the unit circle
        r = [np.cos(phi[n]), np.sin(phi[n])]

        fr = df.Expression('cos(k0L * (rx * x[0] + ry * x[1]))', \
            degree = 1, k0L = k0L, rx = r[0], ry = r[1])

        fi = df.Expression('sin(k0L * (rx * x[0] + ry * x[1]))', \
            degree = 1, k0L = k0L, rx = r[0], ry = r[1])

        A1 = df.as_matrix(((1 - r[0] * r[0], r[0] * r[1]), (0, 0)))
        A2 = df.as_matrix(((0, 0), (r[0] * r[1], 1 - r[1] * r[1])))

        #-----------------------------------------------------------------------
        # Calculating components of far field pattern (real and imaginary part)
        # according to Mischenko: Electromagnetic scattering by Particles and
        # Particle Groups. Cambridge University Press
        #-----------------------------------------------------------------------
        FF_r1[n] = (k0L * k0L) * df.assemble(df.dot((A1 * (permittivity - II)) \
            * (E_r * fr + E_i * fi), e1) * df.dx) / (4 * 3.1415)

        FF_r2[n] = (k0L * k0L) * df.assemble(df.dot((A2 * (permittivity - II)) \
            * (E_r * fr + E_i * fi), e2) * df.dx) / (4 * 3.1415)

        FF_i1[n] = (k0L * k0L) * df.assemble(df.dot((A1 * (permittivity - II)) \
            * (E_i * fr - E_r * fi), e1) * df.dx) / (4 * 3.1415)

        FF_i2[n] = (k0L * k0L) * df.assemble(df.dot((A2 * (permittivity - II)) \
            * (E_i * fr - E_r * fi), e2) * df.dx) / (4 * 3.1415)

        FF[n] = np.sqrt(FF_r1[n] * FF_r1[n] + FF_i1[n] * FF_i1[n] \
            + FF_r2[n] * FF_r2[n] + FF_i2[n] * FF_i2[n])
        #-----------------------------------------------------------------------

    # Write far field pattern to output_folder/ff_mesh_name file
    ofile = open(output_folder + '/ff_' + mesh_name, 'w')
    for m in range(0, FF_n):
        ofile.write('%8.4e %8.4e\n' % (phi[m], FF[m]))

    return phi, FF
#-------------------------------------------------------------------------------


def save_PVD(output_folder, output_name, output_file):
    """Save function output_file to .pvd file format"""
    # Input Variables:
        # output_folder: folder where .h5 file will be store, format folder/
        # mesh_name: name of .h5 file in which mesh is stored
        # output_file: function that will be saved in
        #   'output_folder/Field_mesh_name.pvd' file

    # Output:
        # function output_file will be saved to output_folder/output_file.pvd

    vtkfile = df.File(output_folder + output_name + '.pvd')
    vtkfile << output_file

    return 0
#-------------------------------------------------------------------------------


def save_HDF5(output_folder, output_name, mesh, markers, output_file):
    """Save function output_file and coresponding mesh to .h5 file format"""
    # Input Variables:
        # output_folder: folder where .h5 file will be store
        # output_name: name of the output file
        # mesh: FEniCS mesh function
        # markers: FEniCS subdomains function
        # output_file: function that will be saved in output

    # Output:
        # function output_file will be saved to output_folder/output_file.h5

    # Open file output_folder/output_name.h5 for writing
    hdf = df.HDF5File(mesh.mpi_comm(), output_folder + output_name + '.h5', 'w')

    hdf.write(mesh, output_folder + 'mesh')
    hdf.write(markers, output_folder + 'subdomains')
    hdf.write(output_file, output_folder + 'solution');

    hdf.close()

    return 0
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Main part
#-------------------------------------------------------------------------------

if __name__ == "__main__":

    # Function call: python3 anisotropic2D.py input_folder mesh_name permittivity_file output_folder FF_n
    # ie. python3 anisotropic2D.py mesh anisotropic effective results 72

    # Input parameters
    input_folder = sys.argv[1]
    mesh_name = sys.argv[2]
    permittivity_file = sys.argv[3]
    output_folder = sys.argv[4]
    FF_n = int(sys.argv[5])

    air_permittivity = 1

    # Plane Wave excitation E = p exp(i * k0L * s.x)
    s = [1, 2];    p = [-2, 1];  k0L = 3.141592
    pw_r, pw_i = plane_wave_2D(s, p, k0L)

    # Mesh function
    mesh, markers, permittivity = mesh_anisotropic_2D(input_folder, mesh_name, \
        permittivity_file, air_permittivity)

    # Solver call (solutions are in N1curl FE space)
    E_r, E_i = solver_anisotropic_2D(mesh, permittivity, pw_r, pw_i, k0L)

    # P1 vector and scalar FE space
    V = df.FunctionSpace(mesh, 'P', 1)
    V3 = df.VectorFunctionSpace(mesh, 'P', 1)

    # Project solution from N1Curl to P1 FE space
    EP1_r = df.project(E_r, V3);  EP1_i = df.project(E_i, V3);

    # Output files in PVD (for ParaView) and HDF5 (for later processing) format
    save_PVD(output_folder + '/PVD/', 'Er_' + mesh_name, EP1_r);
    save_PVD(output_folder + '/PVD/', 'Ei_' + mesh_name, EP1_i)

    save_HDF5(output_folder +'/XDMF/', 'Er_' + mesh_name, mesh, markers, EP1_r)
    save_HDF5(output_folder +'/XDMF/', 'Ei_' + mesh_name, mesh, markers, EP1_i)

    # Far field computation
    phi, FF = ff_anisotropic_2D(mesh_name, output_folder, permittivity, \
        k0L, EP1_r, EP1_i, FF_n)
