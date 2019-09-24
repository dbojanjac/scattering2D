import dolfin as df
import numpy as np
from .scattering import Scattering


class IsotropicCoeff(df.Expression):
    # permittivity is dict {int: float}
    # - 1: represents outside material (air_permittivity)
    # - 2: represents material matrix (matrix_permittivity)
    # - 3: represents inner material (patch_permittivity)
    def __init__(self, markers, permittivity, **kvargs):
        self.markers = markers
        self.permittivity = permittivity

    def eval_cell(self, values, x, cell):
        values[0] = self.permittivity[self.markers[cell.index]]


class IsotropicScattering(Scattering):
    def __init__(self, mesh_filename, permittivity_dict, **kvargs):
        super().__init__(mesh_filename, permittivity_dict, **kvargs)
        self.II = 1

        self.permittivity = IsotropicCoeff(self.mesh_markers,
                self.permittivity_dict, degree=0)


    def _get_ff_component(self, k0L, A1, A2, E_r, fr, E_i, fi, e1, e2):
        FF_r1 = (k0L * k0L) * df.assemble((self.permittivity - self.II) *
                df.inner(A1 * (E_r * fr + E_i * fi), e1) * df.dx) / (4 * np.pi)
        FF_r2 = (k0L * k0L) * df.assemble((self.permittivity - self.II) *
                df.inner(A2 * (E_r * fr + E_i * fi), e2) * df.dx) / (4 * np.pi)
        FF_i1 = (k0L * k0L) * df.assemble((self.permittivity - self.II) *
                df.inner(A1 * (E_i * fr - E_r * fi), e1) * df.dx) / (4 * np.pi)
        FF_i2 = (k0L * k0L) * df.assemble((self.permittivity - self.II) *
                df.inner(A2 * (E_i * fr - E_r * fi), e2) * df.dx) / (4 * np.pi)

        return FF_r1 * FF_r1 + FF_i1 * FF_i1 + FF_r2 * FF_r2 + FF_i2 * FF_i2
