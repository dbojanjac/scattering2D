import firedrake as fd

from .scattering import Scattering

class AnisotropicScattering(Scattering):
    def __init__(self, mesh_filename, permittivity_dict, k0L, **kvargs):
        super().__init__(mesh_filename, k0L, **kvargs)
        self.II = fd.as_matrix(((1, 0), (0,1)))
        self.permittivity = fd.Function(fd.TensorFunctionSpace(self.mesh, "DG", 0))

        for (subd_id, epsilon_tensor) in permittivity_dict.items():
            epsilon = fd.as_matrix(epsilon_tensor)
            self.permittivity.interpolate(epsilon,
                                          self.mesh.measure_set("cell", subd_id))

