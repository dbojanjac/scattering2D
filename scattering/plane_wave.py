import firedrake as fd
import numpy as np

def orthogonal(s, p):
    return np.isclose(np.dot(s, p), 0)

def normalize(a):
    return a / np.linalg.norm(a)

class PlaneWave():
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

        self.s = fd.as_vector(s)
        self.p = fd.as_vector(p)

        print("[DEBUG] s = {}".format(s))
        print("[DEBUG] p = {}".format(p))

    #TODO: this should be made abtract and other excitements should implement
    def interpolate(self, mesh, k0L):
        V = fd.VectorFunctionSpace(mesh, "CG", 1)
        x = fd.SpatialCoordinate(mesh)

        pw = fd.interpolate(self.p * fd.exp(1j * k0L * fd.dot(self.s, x)), V)

        return pw
