# FEM based solver for electromagnetic wave scattering in 2D on anisotropic or
# heterogeneous isotropic material based on weak formulatio and FEM

# Function call: python3 isotropic2D.py mesh_folder mesh_name output_folder FF_n
# ie. python3 isotropic2D.py mesh isotropic results 36

# input = domain mesh with subdomain markers in .h5 format
# output = real and imaginary part of total electric field and far field pattern

# Using FEniCS 2017.2.0
from dolfin import *
import numpy as np
import sys


def read_HDF5_file(solution_folder, mesh, solution):
    """Read function solution and coresponding mesh from .h5 file format"""

    # Input Variables:
        # Solutiont_Folder: folder where .h5 file will be store, format folder/
        # mesh: mesh keeping variable
        # solution: name of .h5 file in which mesh is stored

    # Output Variables:
        # u: function read from solution_folder/solution.h5 file in Function Space V
        # mesh: mesh read from solution_folder/solution.h5 file

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), solution_folder + solution, 'r')
    hdf.read(mesh, solution_folder + 'mesh', False)

    V = VectorFunctionSpace(mesh, 'P', 1);   u = Function(V)
    hdf.read(u, solution_folder + 'solution');  hdf.close()

    return u, mesh, V
#-------------------------------------------------------------------------------


def save_PVD(Output_Folder, Output_Name, u):
    """Save function u and coresponding mesh to .pvd file format"""

    # Input Variables:
        # Output_Folder: folder where .h5 file will be store, format folder/
        # mesh_name: name of .h5 file in which mesh is stored
        # Field:
        # u: function that will be saved in 'Output_Folder/Field_mesh_name.pvd' file

    vtkfile = File(Output_Folder + Output_Name + '.pvd')
    vtkfile << u

    return 0
#-------------------------------------------------------------------------------


def save_HDF5(Output_Folder, mesh, mesh_name, Field, u):
    """Save function u and coresponding mesh to .h5 file format"""

    # Input Variables:
        # Output_Folder: folder where .h5 file will be store, format folder/
        # mesh: mesh keeping variable
        # mesh_name: name of .h5 file in which mesh is stored
        # Field:
        # u: function that will be saved in 'Output_Folder/Field_mesh_name.h5' file

    hdf = HDF5File(mesh.mpi_comm(), Output_Folder + Field + '_' + mesh_name + '.h5', 'w')
    hdf.write(mesh, Output_Folder + 'mesh')
    hdf.write(u, Output_Folder + 'solution');   hdf.close()

    return 0
#-------------------------------------------------------------------------------


def mesh_isotropic_2D(mesh_folder, mesh_name, patch_permittivity, matrix_permittivity, air):
    """Read mesh and subdomains, from .h5 mesh file, for heterogeneous isotropic domain"""

    # Input Variables:
        # mesh_folder: mesh in .h5 file format contaning folder
        # mesh_name: name of the mesh containing file (in .h5 format), mesh subdomains
                   # are stored in subdomains part of mesh_folder/mesh_name.h5 file
                   # coefficient 3 represents patch_permittivity,  2 represents sillica and
                   # 1 represents fresh air
        # patch_permittivity: free space permittivity coefficient
        # matrix_permittivity: material matrix permittivity coefficient

    # Output Variables:
        # mesh: mesh read from mesh_folder/Mesh_nNme.h5 file
        # markers: physical subdomains defining function
        # permittivity: domain permittivity

    # Read mesh and subdomain markers from mesh_folder/mesh_name
    #---------------------------------------------------------------------------
    mesh_folder = mesh_folder + '/'

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_folder + mesh_name + '.h5', 'r')

    hdf.read(mesh, mesh_folder + "mesh", False)
    markers = MeshFunction('int', mesh)
    hdf.read(markers, mesh_folder + "subdomains")

    #---------------------------------------------------------------------------
    # Permittivity coefficient for previously defined subdomains
    #---------------------------------------------------------------------------
    class Coeff(Expression):

        def __init__(self, mesh, **kwargs):
            self.markers = markers

        def eval_cell(self, values, x, cell):

            if markers[cell.index] == 3:
                values[0] = patch_permittivity

            if markers[cell.index] == 2:
                values[0] = matrix_permittivity

            else:
                values[0] = air

    # Interpolation to zeroth order polynomial
    permittivity = Coeff(mesh, degree=0)

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


    # Check if polarization is orthogonal to direction of propagation
    if (s[0] * p[0] + s[1] * p[1]) <= 1E-8:

        # make unit vectors from s and p
        s_norm = sqrt(s[0] ** 2 + s[1] ** 2); s[0], s[1] = s[0] / s_norm, s[1] / s_norm;
        p_norm = sqrt(p[0] ** 2 + p[1] ** 2); p[0], p[1] = p[0] / p_norm, p[1] / p_norm

        pw_r = Expression(\
        ('px * cos(k0L * (s_x * x[0] + s_y * x[1]))', \
        'py * cos(k0L * (s_x * x[0] + s_y * x[1]))'), \
        degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

        pw_i = Expression(\
        ('px * sin(k0L * (s_x * x[0] + s_y * x[1]))', \
        'py * sin(k0L * (s_x * x[0] + s_y * x[1]))'), \
        degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

    # Make it orthogonal by ignoring electric field component in the
    # direction of propagation
    else:

        # Subtract from polarization vector projection in the direction of the propagation
        p[0] = p[0] - ((s[0] * p[0] + s[1] * p[1]) / (s[0] * s[0] + s[1] * s[1])) * s[0]
        p[1] = p[1] - ((s[0] * p[0] + s[1] * p[1]) / (s[0] * s[0] + s[1] * s[1])) * s[1]

        # make unit vectors from s and p
        s_norm = sqrt(s[0] ** 2 + s[1] ** 2); s_x, s_y = s[0] / s_norm, s[1] / s_norm;
        p_norm = sqrt(p[0] ** 2 + p[1] ** 2); px, py = p[0] / p_norm, p[1] / p_norm

        pw_r = Expression(\
        ('px * cos(k0L * (s_x * x[0] + s_y * x[1]))', \
        'py * cos(k0L * (s_x * x[0] + s_y * x[1]))'), \
        degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

        pw_i = Expression(\
        ('px * sin(k0L * (s_x * x[0] + s_y * x[1]))', \
        'py * sin(k0L * (s_x * x[0] + s_y * x[1]))'), \
        degree=1, px = p[0], py = p[1], s_x = s[0], s_y = s[1], k0L = k0L)

    return pw_r, pw_i
#-------------------------------------------------------------------------------


def solver_isotropic_2D(mesh, permittivity, pw_r, pw_i, k0L):
    """Electromagnetic scattering solver based on FEM"""

    # Input Variables:
        # mesh: mesh keeping variable
        # permittivity: permittivity variable (isotropic function)
        # pw_r: real part of incoming plane wave
        # pw_i: imaginary part of incoming plane wave
        # k0L: dimensionless parameter describing wave vector length

    # Output Variables:
        # E_r: real part of total electric field
        # E_i: imaginary part of total electric field

    #---------------------------------------------------------------------------
    # Finite Element function spaces (Nedelec N1curl space), mixed formulation
    #---------------------------------------------------------------------------
    NED = FiniteElement('N1curl', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, 'N1curl', 1)
    W = FunctionSpace(mesh, NED * NED)

    #---------------------------------------------------------------------------
    # Weak formulation
    #---------------------------------------------------------------------------
    Es_r, Es_i = TrialFunctions(W)
    v_r, v_i = TestFunctions(W)
    n = FacetNormal(mesh)

    a_r = inner(curl(Es_r), curl(v_r)) * dx - k0L * k0L * inner(permittivity * Es_r, v_r) * dx + \
        k0L * (inner(n, Es_i) * inner(n, v_r) - inner(Es_i, v_r)) * ds

    a_i = inner(curl(Es_i), curl(v_i)) * dx - k0L * k0L * inner(permittivity * Es_i, v_i) * dx - \
        k0L * (inner(n, Es_r) * inner(n, v_i) - inner(Es_r, v_i)) * ds

    L_r = - k0L * k0L * inner((1 - permittivity) * pw_r, v_r) * dx
    L_i = - k0L * k0L * inner((1 - permittivity) * pw_i, v_i) * dx

    # Final variational form
    F = a_r + a_i - L_r - L_i

    # Splitting the variational form into LHS and RHS
    a, L = lhs(F), rhs(F)

    # Solution Function
    es = Function(W); Es = es.vector()

    # Assemble RHS, LHS and solve the system
    A = assemble(a);    b = assemble(L);    solve(A, Es, b)

    # Split Solution into real and imaginary part
    Es_r, Es_i = es.split()

    # Interpolate incoming plane wave into N1Curl Function Space
    EI_r = interpolate(pw_r, V); EI_i = interpolate(pw_i, V)

    # Total Field = Incident Field + Scattered Field
    E_r = Es_r + EI_r;  E_i = Es_i + EI_i

    return E_r, E_i
#-------------------------------------------------------------------------------


def ff_isotropic_2D(permittivity, k0L, e_r, e_i, m):
    """Far Field calculator"""

    # Input Variables:
        # mesh: mesh keeping variable
        # permittivity: permittivity
        # pw_r: real part of incoming plane wave
        # pw_i: imaginary part of incoming plane wave
        # k0L: name of .h5 file in which mesh is stored

    # Output Variables:
        # phi: angle list
        # ff: far field value

    step = 1 / float(m - 1)

    phi = np.linspace(step / 2, 2 * 3.1415 - step / 2, num = m)

    rez1r = [0] * m;   rez2r = [0] * m; rez1i = [0] * m;    rez2i = [0] * m
    ff21 = [0] * m;   ff22 = [0] * m;     ff = [0] * m;

    for n in range (0, m):

        r = [np.cos(phi[n]), np.sin(phi[n])]

        fr = Expression('cos(k0L * (rx * x[0] + ry * x[1]))', degree = 1, k0L = k0L, rx = r[0], ry = r[1])
        fi = Expression('sin(k0L * (rx * x[0] + ry * x[1]))', degree = 1, k0L = k0L, rx = r[0], ry = r[1])

        A1 = as_matrix(((1 - r[0] * r[0], r[0] * r[1]), (0, 0)))
        A2 = as_matrix(((0, 0), (r[0] * r[1], 1 - r[1] * r[1])))

        e1 = as_vector([1, 0]);   e2 = as_vector([0, 1])

        rez1r[n] = (k0L * k0L) * assemble((permittivity - 1) * dot(A1 * (e_r * fr + e_i * fi), e1) * dx) / (4 * 3.1415)
        rez2r[n] = (k0L * k0L) * assemble((permittivity - 1) * dot(A2 * (e_r * fr + e_i * fi), e2) * dx) / (4 * 3.1415)

        rez1i[n] = (k0L * k0L) * assemble((permittivity - 1) * dot(A1 * (e_i * fr - e_r * fi), e1) * dx) / (4 * 3.1415)
        rez2i[n] = (k0L * k0L) * assemble((permittivity - 1) * dot(A2 * (e_i * fr - e_r * fi), e2) * dx) / (4 * 3.1415)

        ff21[n] = rez1r[n] * rez1r[n] + rez1i[n] * rez1i[n]
        ff22[n] = rez2r[n] * rez2r[n] + rez2i[n] * rez2i[n]

        ff[n] = np.sqrt(ff21[n] + ff22[n])


    return phi, ff
#-------------------------------------------------------------------------------



if __name__ == "__main__":

    # Function call: python3 isotropic2D.py mesh_folder mesh_name output_folder FF_n
    # ie. python3 isotropic2D.py mesh isotropic results 36

    mesh_folder = sys.argv[1]
    mesh_name = sys.argv[2]
    output_folder = sys.argv[3]
    FF_n = int(sys.argv[4])

    # Domain defining permittivity coefficients
    patch_permittivity = 1
    matrix_permittivity = 11.7
    air = 1

    # Plane Wave excitation E = p exp(i * k0L * s.x)
    s = [1, 2];    p = [-2, 1];  k0L = 3.141592653589793

    pw_r, pw_i = plane_wave_2D(s, p, k0L)

    # Mesh function
    mesh, markers, permittivity = mesh_isotropic_2D(mesh_folder, mesh_name, patch_permittivity, matrix_permittivity, air)

    # Solver call
    E_r, E_i = solver_isotropic_2D(mesh, permittivity, pw_r, pw_i, k0L)

    # P1 FE space
    V3 = VectorFunctionSpace(mesh, 'P', 1); V = FunctionSpace(mesh, 'P', 1)

    # Project solution from N1Curl to P1 FE space
    EP1_r = project(E_r, V3);  EP1_i = project(E_i, V3);

    # Output files in PVD (for ParaView) and HDF5 (for later processing) format
    save_PVD(output_folder + '/PVD/', 'E_r', EP1_r);
    save_PVD(output_folder + '/PVD/', 'E_i', EP1_i)
    save_HDF5(output_folder +'/XDMF/', mesh, mesh_name, 'E_r', EP1_r)
    save_HDF5(output_folder +'/XDMF/', mesh, mesh_name, 'E_i', EP1_i)

    # Far field computation
    phi, ff = ff_isotropic_2D(permittivity, k0L, EP1_r, EP1_i, FF_n)
    ofile = open('ff_' + mesh_name, 'w')
    for m in range(0, FF_n):
        ofile.write('%8.4e %8.4e\n' % (phi[m], ff[m]))
