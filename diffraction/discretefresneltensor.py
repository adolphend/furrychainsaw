import numpy as np
import logging as lg
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiscreteFresnel(nn.Module):
    def __init__(self, wavelength, DeltaX0, DeltaY0, DeltaXZ, DeltaYZ, Nx, Ny):
        self.wavelength = wavelength
        self.DeltaX0 = DeltaX0
        self.DeltaY0 = DeltaY0
        self.DeltaXZ = DeltaXZ
        self.DeltaYZ = DeltaYZ
        self.Nx = Nx
        self.Ny = Ny

    def computeMu(self, z):
        self.muz = torch.exp(1j * 2 * torch.pi / self.wavelength) / (1j * self.wavelength * z) * self.DeltaX0 * self.DeltaY0
        return self.muz

    def computeCyz(self, z):
        s, s_prime = np.indices((self.Ny, self.Ny))
        s, s_prime = torch.from_numpy(s).to(device), torch.from_numpy(s_prime).to(device)
        s, s_prime = (s - self.Ny/2) * self.DeltaYZ, (s_prime - self.Ny/2) * self.DeltaY0
        self.Cyz = torch.exp(1j * torch.pi /(self.wavelength * z) * (s - s_prime)**2)
        return self.Cyz

    def computeCxz(self, z):
        s, s_prime = np.indices((self.Nx, self.Nx))
        s, s_prime = torch.from_numpy(s), torch.from_numpy(s_prime)
        s, s_prime = (s - self.Nx/2) * self.DeltaXZ, (s_prime - self.Nx/2) * self.DeltaX0
        self.Cxz = torch.exp(1j * torch.pi /(self.wavelength * z) * (s - s_prime)**2)
        return self.Cxz

    def computeGD(self, z):
        if z < 10 * self.wavelength:
            lg.warning("look at discrete fresnel approximation")
        Muz = self.computeMu(z)
        Cyz = self.computeCyz(z)
        Cxz = self.computeCxz(z)
        return Muz * torch.kron(Cyz, Cxz)

    def __str__(self):
        return "lambda: " + str(self.wavelength) + ", Nx: " + str(self.Nx)
