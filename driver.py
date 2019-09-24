from scattering import IsotropicScattering, AnisotropicScattering, plane_wave
import dolfin as df
import numpy as np
import argparse

def log_args(args, subdomains):
    if args.eps_tensor:
        print ("Running anisotropic problem with mesh {} and permittivity \
                tensor {}".format(args.mesh, subdomains))
    else:
        print ("Running isotropic problem with mesh {} and permittivity \
                scalars {}".format(args.mesh, subdomains))

    print("Generating far field output in {} with {} samples.".format(
        args.FF_n, args.output))


def log_solutions(E_r, E_i, mesh, anisotropic):
    # P1 vector and scalar FE space
    V = df.FunctionSpace(mesh, 'P', 1)
    V3 = df.VectorFunctionSpace(mesh, 'P', 1)

    # Project solution from N1Curl to P1 FE space
    EP1_r = df.project(E_r, V3);  EP1_i = df.project(E_i, V3);

    if anisotropic:
        E_r_out = df.File("E_r_anisotropic.pvd")
        E_i_out = df.File("E_i_anisotropic.pvd")
    else:
        E_r_out = df.File("E_r_isotropic.pvd")
        E_i_out = df.File("E_i_isotropic.pvd")

    E_r_out << EP1_r
    E_i_out << EP1_i

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "FEM based solver for " +
        "electromagnetic wave scattering in 2D on anisotropic or isotropic " +
        "materials.")

    parser.add_argument("permittivity", nargs='+',
                        help="Permittivity accordingly to subdomain_id",
                        type=float)

    parser.add_argument("-o", "--output", help="Folder for outputing results")
    parser.add_argument("-e", "--eps-tensor", help="Epsilon tensor filename.")
    parser.add_argument("-n", "--far-field-num", help="Number of dots to \
            evaluate far field", dest="FF_n", type=int)
    # don't use -m because that is reserved and there is no bash autocomplete
    parser.add_argument("--mesh", help="Mesh filename")

    args, petsc_args = parser.parse_known_args()

    print("Passed args: {}".format(args))
    print ("Using PETSc args {}".format(petsc_args))

    s = [0, 1]
    p = [1, 0]
    k0L = np.pi

    if not args.FF_n:
        args.FF_n = 10

    if not args.output:
        if args.eps_tensor:
            args.output = "out_anisotropic.txt"
        else:
            args.output = "out_isotropic.txt"

    if args.eps_tensor:
        subdomains = {3: [args.permittivity[0], 0, 0, args.permittivity[0]]}
        with open(args.eps_tensor, "r") as f:
            eps_tensor = []
            for line in f:
                eps_tensor.extend(line.split())
        eps_tensor = [float(i) for i in eps_tensor]
        subdomains.update({1: eps_tensor})

        log_args(args, subdomains)
        scattering = AnisotropicScattering(args.mesh, subdomains)
        pw_r, pw_i = plane_wave(s, p, k0L)
        E_r, E_i = scattering.solve(pw_r, pw_i, k0L)

        log_solutions(E_r, E_i, scattering.mesh, args.eps_tensor)
        scattering.get_far_field(args.FF_n, k0L, E_r, E_i, output_file=args.output)
    else:
        subdomains = {i: args.permittivity[i-1]
                for i in range(1, len(args.permittivity)+1)}

        log_args(args, subdomains)
        scattering = IsotropicScattering(args.mesh, subdomains)
        pw_r, pw_i = plane_wave(s, p, k0L)
        E_r, E_i = scattering.solve(pw_r, pw_i, k0L)
        log_solutions(E_r, E_i, scattering.mesh, args.eps_tensor)

        scattering.get_far_field(args.FF_n, k0L, E_r, E_i, output_file=args.output)


