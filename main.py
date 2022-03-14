from diffraction import GaussianBeam, DiscreteFresnel
from tools import generateFxFy, heatmapfile
import numpy as np
from scipy.linalg import hadamard

np.random.seed(1234)

def main():
    w0 = .5 
    N = 32
    z = 10
    wavelength = 0.856
    Delta = .5
    f = 100
    l = 10
    X = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            X[i, j] = int(((((i - N/2) ** 2 + (j - N/2) ** 2) < (N * .5) ** 2) and
                          not (((i - N/2) ** 2 + (j - N/2) ** 2) < (N * .4) ** 2))  or
                          (((i - N/2) ** 2 + (j - N/2) ** 2) < (N * .08) ** 2) or 
                          (((i+j)% 8 < 2)))
        string = [" " if i else "*" for i in X[i, :]]
        print(" ".join(string))
    heatmapfile(X, "X.jpg")
    gb = GaussianBeam(wavelength, w0)
    X, Y = generateFxFy(N, N, Delta, Delta)
    e, p = gb.computeE(X, Y, z=(f+l), f=f)
    p = np.absolute(p) * np.exp(-1j * (np.angle(p) - np.mean(np.angle(p))))
    E = e * p
    heatmapfile(np.absolute(E), "amplitude.jpg")
    heatmapfile(np.angle(E), "phase.jpg")
    df = DiscreteFresnel(wavelength, Delta, Delta, Delta, Delta, N, N)
    dms = df.computeGD(z)
    dsr = df.computeGD(z+l)
    mask = np.random.binomial(1, 0.5, N*N)
    #mask = hadamard(N).reshape(-1)
    maskTHz = mask * E.reshape(-1)
    heatmapfile(np.absolute(mask).reshape((N, N)), "maskOPt%f.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(mask).reshape((N, N)), "maskOptPhse%f.jpg"%z, cmap="Blues")
    heatmapfile(np.absolute(maskTHz).reshape((N,N)), "maskTHz%f.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(maskTHz).reshape((N,N)), "maskTHzPhse%f.jpg"%z, cmap="Blues")
    maskP = np.dot(dms, maskTHz)
    heatmapfile(np.absolute(maskP).reshape((N,N)), "maskPAbsolute%f.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(maskP).reshape((N,N)), "maskPPhse%f.jpg"%z, cmap="Blues")
    maskPImage = maskP * X.reshape((N*N))
    maskRefrection = np.dot(dsr, maskPImage)
    heatmapfile(np.absolute(maskRefrection).reshape((N, N)), "maskRefrectionAbsolute%f.jpg"%z)
    heatmapfile(np.angle(maskRefrection).reshape((N,N)), "maskRefrectionPhse%f.jpg"%z)

    maskCollection = 1j * gb.k / (2 * np.pi * f) * np.exp(-1j * gb.k * f)
    
    yp = np.absolute(np.sum(maskCollection * maskRefrection))
    
    e_i = np.dot(np.dot(np.conjugate(np.diag(np.dot(dms, maskTHz))), np.dot(np.transpose(np.conjugate(maskCollection * dsr)), np.ones((N*N, 1)))).T, X.reshape((N*N, 1)))
    yn = np.absolute(e_i)

    print(yp ** 2 / 2, yn  ** 2 / 2)
    sys = ThzImaging(wavelength, w0, f, l, f, N, N, Delta, Delta)
    sys.generateTtoM()
    sys.generateDS(z, Delta, Delta)
    Ai = sys.computeTildeAi(maskTHz, sys.Dsr, sys.Dms)
    y = sys.computeYi(Ai, X)
    print(sys)
    A, y = sys.generateMeasurement(z, Delta, Delta, X)
    TildeX = np.dot(np.linalg.pinv(A), (y) ** 0.5 * 0.5)
    heatmapfile(np.absolute(TildeX).reshape((N, N)), "TildeX.jpg")
    from tools import CoPRAM
    x = CoPRAM(y ** 0.5 * 0.5, A, 500, 10, 0.1, 0.1, X.reshape(-1))
    heatmapfile(np.absolute(x.reshape((N, N)), "x.jpg"))
    from cosamp import cosamp
    CosampX = cosamp.cosamp(A, y, 500)
    heatmapfile(np.absolute(TildeX).reshape((N, N)), "CosampX.jpg")


class ThzImaging():
    def __init__(self,wavelength, w0,Ztl, Zlm, focal, Nx, Ny, DeltaXM, DeltaYM):
        """
        Args:
            - wavelength(float): wavelength of the source
            - w0: beamwidth of the source
            - Ztl: distance between the lens and the transmitter
            - Zlm: distance between the lens and the spatial modulator
            - focal: lenses as focal lens
            - Nx: number of pixel in width
            - Ny: number of pixel in height
            - DeltaXM: step size width for the mask
            - DeltaYM: step size height for the mask
        """
        self.wavelength = wavelength
        self.w0 = w0
        self.Ztl = Ztl
        self.Zlm = Zlm
        self.focal = focal
        self.Nx = Nx
        self.Ny = Ny
        self.DeltaXM = DeltaXM
        self.DeltaYM = DeltaYM

    def generateTtoM(self, rotate=True):
        self.GaussianBeam = GaussianBeam(self.wavelength, self.w0)
        X, Y = generateFxFy(self.Nx, self.Ny, self.DeltaXM, self.DeltaYM)
        z = self.Ztl + self.Zlm
        E, P = self.GaussianBeam.computeE(X, Y, z=z, f=self.focal)
        if rotate:
            P = np.absolute(P) * np.exp(1j * (np.angle(P) - np.mean(np.angle(P))))
        self.E = E * P

    def generateMask(self, cathegory='bernoulli', p=0.5):
        if cathegory == 'bernoulli':
            OptMask = np.random.binomial(1, p, self.Nx * self.Ny)
        ThzMask = self.E.reshape(-1) * OptMask
        return OptMask, ThzMask

    def generateD(self, DeltaX0, DeltaY0, DeltaXZ, DeltaYZ, z):
        Diffr = DiscreteFresnel(self.wavelength, DeltaX0, DeltaY0, DeltaXZ, DeltaYZ, self.Nx, self.Ny)
        DF = Diffr.computeGD(z)
        return DF

    def generateDS(self, z, DeltaXS, DeltaYS):
        self.z = z
        self.DeltaXS = DeltaXS
        self.DeltaYS = DeltaYS
        self.zl = z + self.Zlm
        self.Dms = self.generateD(self.DeltaXM, self.DeltaYM, self.DeltaXS, self.DeltaYS, self.z)
        self.Dsl = self.generateD(self.DeltaXS, self.DeltaYS, self.DeltaXM, self.DeltaYM, self.zl)
        self.Dsr = self.Dsl * np.exp(-2j * np.pi * self.focal / self.wavelength) * 1j /(self.wavelength * self.focal)
        
    def computeTildeAi(self, mask, Dsr, Dms):
        return np.dot(np.conjugate(np.diag(np.dot(Dms, mask))), np.dot(np.conjugate(np.transpose(Dsr)), np.ones((self.Nx * self.Ny, 1))))

    def computeYi(self, Ai, X):
        return np.absolute(np.dot(Ai.T, X.reshape(-1))) ** 2 / 2

    def generateMeasurement(self, z, DeltaXS, DeltaYS, X, mask="bernoulli", m=None):
        if m is None:
            m = self.Nx * self.Ny
        self.X = X
        y = np.zeros(m)
        A = np.zeros((m, self.Nx * self.Ny), dtype=np.complex128)
        self.generateTtoM()
        self.generateDS(z, DeltaXS, DeltaYS)
        for i in range(m):
            _, mask = self.generateMask()
            Ai = self.computeTildeAi(mask, self.Dsr, self.Dms)
            yi = self.computeYi(Ai, X)
            y[i] = yi
            A[i, :] = Ai.T
        return A, y

    def __str__(self):
        print()
        print(  "               \\\\       ||                   ||||      \n",
                "GaussianBeam =>)) Lens || Spatial Modulator |||| Scene\n",
                "              //       ||                   ||||      \n",
                "              z_ts     z_lm                 z_ms")
        print(  "")
        print(  "                ===>==>      \\\\            \n",
                "SceneRefrection ===>==> Lens ))>> Receiver\n",
                "               ===>==>     //            \n",
                "                  z_sl         focal        ")
        string= "Gaussian Beam:\n"+\
               "______________" +\
               "\nwavelength: " + str(self.wavelength) +\
               "\nbeamwidth: " + str(self.w0) +\
               "\ndistance between transmitter and lens: " + str(self.Ztl) +\
               "\ndistance between lens and spatial modulator: " + str(self.Zlm) + "\n"\
               "\n Spatial Optical Modulator:\n"+\
               "___________________________" +\
               "\nNumber of pixel Nx: " + str(self.Nx) +\
               "\nNumber of pixel Ny: " + str(self.Ny) +\
               "\nsize of pixel DeltaX: " + str(self.DeltaXM) +\
               "\nsize of pixel DeltaY: " + str(self.DeltaYM) +"\n"
        try:
            string = string +\
                    "\nScene Information Details:" +\
                    "\n__________________________" +\
                    "\nNumber of pixel Nx: " + str(self.Nx) +\
                    "\nNumber of pixel Ny: " + str(self.Ny) +\
                    "\nsize of pixel DeltaX: " + str(self.DeltaXS) +\
                    "\nsize of pixel DeltaY: " + str(self.DeltaYS) +\
                    "\ndistance between spatial modulator and scene: " + str(self.z) +"\n"
        except:
            pass
        return string

main()
