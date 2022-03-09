import numpy as np
import logging as lg

class GaussianBeam():
    def __init__(self, wavelength, w_0, z=None):
        """
        """
        self.wavelength = wavelength
        self.w_0 = w_0
        self.k = 2 * np.pi / self.wavelength
        self.zc = np.pi * self.w_0 ** 2 / self.wavelength 

    def computeW(self, z=None):
        self.__updateZ(z)
        self.w = self.w_0 * np.sqrt(1 + (self.z / self.zc) ** 2)
        return self.w

    def computeR(self, z=None):
        self.__updateZ(z)
        self.R = self.z + self.zc ** 2 / self.z
        return self.R

    def computePhi0(self, z=None):
        self.__updateZ(z)
        self.Phi0 = np.arctan(self.z/self.zc)
        return self.Phi0

    def computeE(self, x, y, z=None, f=None):
        r = self.compute_r(x, y, z)
        R = self.computeR(z)
        w = self.computeW(z)
        Phi0 = self.computePhi0(z)
        w0 = self.w_0
        k = self.k
        E_A = w0 / w * np.exp(- (r/w) ** 2)
        Phi = k * z + k * r ** 2 / (2 * R) - Phi0
        if f:
            Phi = Phi - k * r ** 2 / (2 * f)
        return E_A, np.exp(1j * Phi)

    def compute_r(self, x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    def __updateZ(self, z):
        if z:
            self.z = z
        if self.z is None:
            lg.warning("Remember to set your z, mate")

    def __str__(self):
        return "lambda: " + str(self.wavelength) + " w_0: " + str(self.w_0)

