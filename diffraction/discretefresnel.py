import numpy as np
import logging as lg

class DiscreteFresnel():
    def __init__(self, wavelength, DeltaX0, DeltaY0, DeltaXZ, DeltaYZ, Nx, Ny):
        self.wavelength = wavelength
        self.DeltaX0 = DeltaX0
        self.DeltaY0 = DeltaY0
        self.DeltaXZ = DeltaXZ
        self.DeltaYZ = DeltaYZ
        self.Nx = Nx
        self.Ny = Ny

    def computeMu(self, z):
        self.muz = np.exp(1j * 2 * np.pi / self.wavelength) / (1j * self.wavelength* z) * self.DeltaX0 * self.DeltaY0
        return self.muz

    def computeCyz(self, z):
        s, s_prime = np.indices((self.Ny, self.Ny))
        s, s_prime = (s - self.Ny/2) * self.DeltaYZ, (s_prime - self.Ny/2) * self.DeltaY0
        self.Cyz = np.exp(1j * np.pi /(self.wavelength * z) * (s - s_prime)**2)
        return self.Cyz

    def computeCxz(self, z):
        s, s_prime = np.indices((self.Nx, self.Nx))
        s, s_prime = (s - self.Nx/2) * self.DeltaXZ, (s_prime - self.Nx/2) * self.DeltaX0
        self.Cxz = np.exp(1j * np.pi /(self.wavelength * z) * (s - s_prime)**2)
        return self.Cxz

    def computeGD(self, z):
        if z < 10 * self.wavelength:
            lg.warning("look at discrete fresnel approximation")
        Muz = self.computeMu(z)
        Cyz = self.computeCyz(z)
        Cxz = self.computeCxz(z)
        return np.dot(Muz, np.kron(Cyz, Cxz))

    def __str__(self):
        return "lambda: " + str(self.wavelength) + ", Nx: " + str(self.Nx)
