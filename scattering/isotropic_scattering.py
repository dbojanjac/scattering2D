import firedrake as fd
from .scattering import Scattering


class IsotropicScattering(Scattering):
    def __init__(self, mesh, permittivity_dict, k0L, **kvargs):
        super().__init__(mesh, k0L, **kvargs)
        self.II = 1
        self.permittivity = fd.Function(fd.FunctionSpace(self.mesh, "DG", 0))

        for (subd_id, epsilon) in permittivity_dict.items():
            epsilon_const = fd.Constant(epsilon)
            self.permittivity.interpolate(epsilon_const, self.mesh.measure_set("cell", subd_id))
