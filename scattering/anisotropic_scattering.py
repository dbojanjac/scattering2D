import numpy as np
from firedrake import as_matrix, Function, TensorFunctionSpace

from .scattering import Scattering

class AnisotropicScattering(Scattering):
    """
    Derived class for AnisotropicScattering

    Computation using this object models material as an anisotropic scatterer. Permittivity of
    each material is modeled using tensor.
    """
    def __init__(self, mesh, permittivity_dict, k0L, **kwargs):
        super().__init__(mesh, k0L, **kwargs)
        # identity here is matrix
        self.II = as_matrix(np.identity(self.mesh.topological_dimension()))
        self.permittivity = Function(TensorFunctionSpace(self.mesh, "DG", 0))

        for (subd_id, epsilon_tensor) in permittivity_dict.items():
            self.permittivity.interpolate(as_matrix(epsilon_tensor),
                                          self.mesh.measure_set("cell", subd_id))

