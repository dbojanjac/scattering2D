import firedrake as fd
from .scattering import Scattering


class IsotropicScattering(Scattering):
    def __init__(self, mesh_filename, permittivity_dict, k0L, **kvargs):
        super().__init__(mesh_filename, k0L, **kvargs)
        self.II = 1
        self.permittivity = fd.Function(fd.FunctionSpace(self.mesh, "DG", 0))

        for (subdomain_id, epsilon) in permittivity_dict.items():
            self.permittivity.interpolate(fd.Constant(epsilon),
                                          self.mesh.measure_set("cell", subdomain_id))
