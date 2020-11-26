import firedrake as fd
from .scattering import Scattering


class IsotropicScattering(Scattering):
    """
    Derived class for IsotropicScattering

    Computation using this object models material as in isotropic scatterer. Permittivity of
    each material is modeled using one real number.
    """
    def __init__(self, mesh, permittivity_dict, k0L, **kwargs):
        super().__init__(mesh, k0L, **kwargs)
        # identity for isotropic problem is just 1
        self.II = 1
        # permittivity is DG0 function that changes between each subdomain
        self.permittivity = fd.Function(fd.FunctionSpace(self.mesh, "DG", 0))

        for (subd_id, epsilon) in permittivity_dict.items():
            self.permittivity.interpolate(fd.Constant(epsilon),
                                          self.mesh.measure_set("cell", subd_id))
