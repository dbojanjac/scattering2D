import numpy as np
from abc import ABC, abstractmethod

from firedrake import VectorFunctionSpace, SpatialCoordinate, interpolate, dot, exp, as_vector

from .scattering import ExcitationBase

def orthogonal(s, p):
    """Return true if np.arrays that represent vectors are orthogonal"""
    return np.isclose(np.dot(s, p), 0)

def normalize(a):
    """Normalize vector a represented as np.array"""
    return a / np.linalg.norm(a)

class PlaneWave(ExcitationBase):
    """
    Plane wave excitation

    We define plane wave as:
        E_i(x) = \hat{p} * exp(i k0L * \hat{s} \cdot x)
    """

    def __init__(self, s, p):
        # If vectors aren't orthogonal, make them orthogonal by ignoring
        # electric field component in the direction of propagation
        if not orthogonal(s, p):
            # Subtract from polarization vector projection in the direction
            # of the propagation
            p = p - (np.dot(s, p) / np.dot(s, s)) * s

        # make unit vectors from s and p
        s = normalize(s)
        p = normalize(p)

        self.s = as_vector(s)
        self.p = as_vector(p)

    def interpolate(self, mesh, *args):
        """Interpolate plane wave on the mesh"""
        (k0L,) = args
        V = VectorFunctionSpace(mesh, "CG", 1)
        x = SpatialCoordinate(mesh)

        return interpolate(self.p * exp(1j * k0L * dot(self.s, x)), V)
