from diffraction import GaussianBeam, DiscreteFresnel
from tools import generateFxFy, heatmapfile
import numpy as np
from scipy.linalg import hadamard
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


np.random.seed(1234)

def main():
    w0 = .5 
    N = 32
    wavelength = 0.856
    Delta0 = .5
    DeltaZ = Delta0 / 2
    f = 100
    l = 10
    X = np.zeros((N, N))
    z = Delta0 * DeltaZ * N / wavelength * np.sqrt(10) * 2
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
    stupidX, stupidY = generateFxFy(N, N, Delta0, Delta0)
    e, p = gb.computeE(stupidX, stupidY, z=(f+l), f=f)
    image = np.zeros((1000, 1000), dtype=np.uint8)
    image[:256, :256] = cv2.resize(np.asarray(X * 255, dtype=np.uint8), (256, 256))
    video = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc('m','j','p', 'g'), 15, (1000,1000), 0)
    for i in range(1000):
        imagewrite = cv2.putText(image, 'wavelength: ' + str(i), (500,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 1, cv2.LINE_AA) 
        video.write(imagewrite)
        
    video.release()
    #p = np.absolute(p) * np.exp(-1j * (np.angle(p) - np.mean(np.angle(p))))
    E = e * p
    heatmapfile(np.absolute(E), "amplitude.jpg")
    heatmapfile(np.angle(E), "phase.jpg")
    df = DiscreteFresnel(wavelength, Delta0, Delta0, DeltaZ, DeltaZ, N, N)
    dms = df.computeGD(z)
    df = DiscreteFresnel(wavelength, DeltaZ, DeltaZ, Delta0, Delta0, N, N)
    dsr = df.computeGD(z+l)
    mask = np.random.binomial(1, 0.5, N*N)
    #mask = hadamard(N * N)[random.randint(N*N)]
    maskTHz = mask * E.reshape(-1)
    os.makedirs("%f"%z, exist_ok=True)
    heatmapfile(np.absolute(mask).reshape((N, N)), "%f/maskOPt.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(mask).reshape((N, N)), "%f/maskOptPhse.jpg"%z, cmap="Blues")
    heatmapfile(np.absolute(maskTHz).reshape((N,N)), "%f/maskTHz.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(maskTHz).reshape((N,N)), "%f/maskTHzPhse.jpg"%z, cmap="Blues")
    maskP = np.dot(dms, maskTHz)
    heatmapfile(np.absolute(maskP).reshape((N,N)), "%f/maskPAbsolute.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(maskP).reshape((N,N)), "%f/maskPPhse.jpg"%z, cmap="Blues")
    maskPImage = maskP * X.reshape((N*N))
    maskRefrection = np.dot(dsr, maskPImage)
    heatmapfile(np.absolute(maskRefrection).reshape((N, N)), "%f/maskRefrectionAbsolute.jpg"%z, cmap="Blues")
    heatmapfile(np.angle(maskRefrection).reshape((N,N)), "%f/maskRefrectionPhse.jpg"%z, cmap="Blues")

    maskCollection = 1j * gb.k / (2 * np.pi * f) * np.exp(-1j * gb.k * f)
    
    yp = np.absolute(np.sum(maskCollection * maskRefrection))
    
    e_i = np.dot(np.dot(np.conjugate(np.diag(np.dot(dms, maskTHz))), np.dot(np.transpose(np.conjugate(maskCollection * dsr)), np.ones((N*N, 1)))).T, X.reshape((N*N, 1)))
    yn = np.absolute(e_i)

    print(yp ** 2 / 2, yn  ** 2 / 2)
    sys = ThzImaging(wavelength, w0, f, l, f, N, N, Delta0, Delta0)
    sys.generateTtoM()
    sys.generateDS(z, DeltaZ, DeltaZ)
    Ai = sys.computeTildeAi(maskTHz, sys.Dsr, sys.Dms)
    y = sys.computeYi(Ai, X)
    print(sys)
    A, y = sys.generateMeasurement(z, DeltaZ, DeltaZ, X)
    TildeX = np.dot(np.linalg.pinv(A), (y) ** 0.5 * 0.5)
    heatmapfile(np.absolute(TildeX).reshape((N, N)), "TildeX.jpg")
    from tools import CoPRAM
    x = CoPRAM(y ** 0.5 * 0.5, A, 500, 10, 0.1, 0.1, X.reshape(-1))
    heatmapfile(np.absolute(x.reshape((N, N))), "xamplitude.jpg")
    heatmapfile(np.angle(x.reshape((N,N))), "xphase.jpg")
    from cosamp import cosamp
    CosampX = cosamp.cosamp(A, y, 500)
    heatmapfile(np.absolute(TildeX).reshape((N, N)), "CosampX.jpg")

def deltaDistance(z0, zn, D0, DZ, n=100):
    for i in range(n):
        for j in range(n):
            z = (zn - z0) / n * i
            Delta = (DZ - D0) / n * j
            df = DiscreteFresnel(wavelength, Delta, Delta, Delta, Delta, N, N)
            df.computeGD(z)
            maskP = np.dot(df, maskTHz)

def analysisMask():
    z = 100
    Delta = 0.5
    N = 32
    f = 100
    l = 10
    fl = f + l
    wavelength = 0.856
    w0 = 1
    if stupidTest: N=4
    gb = GaussianBeam(wavelength, w0)
    X, Y = generateFxFy(N, N, Delta, Delta)
    e, p = gb.computeE(X, Y, fl, f)
    e = e * p
    masks = hadamard(N*N)
    masksThz = []
    df = DiscreteFresnel(wavelength, Delta, Delta, Delta, Delta, N, N)
    dms = df.computeGD(z)
    title = "mask projection z: %.1f mm, Delta: %.1f mm, N: %d"%(z,Delta,N)
    fig, axis = plt.subplots(2,2)
    fig.suptitle(title)
    fig.tight_layout()
    def init():
        sns.heatmap(np.zeros((N,N)), ax=axis[0,0], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[0,1], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[1,0], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[1,1], cbar=False)
    def heatmap(i):
        [[(i.clear(), i.set_axis_off()) for i in ax]for ax in axis]
        mask = masks[:, i]
        maskTHz = mask * e.reshape(-1)
        proj = np.dot(dms, maskTHz)
        maskAmplitude = np.absolute(maskTHz)
        maskPhase = np.angle(maskTHz)
        projAmplitude = np.absolute(proj)
        projPhase = np.angle(proj)
        sns.heatmap(maskAmplitude.reshape((N,N)), ax=axis[0,0], cbar=False)
        axis[0,0].set_title("Mask amplitude")
        sns.heatmap(maskPhase.reshape((N,N)), ax=axis[0,1], cbar=False)
        axis[0,1].set_title("Mask phase")
        sns.heatmap(projAmplitude.reshape((N,N)), ax= axis[1,0], cbar=False)
        axis[1,0].set_title("Projection Amplitude")
        sns.heatmap(projPhase.reshape((N,N)), ax=axis[1,1], cbar=False)
        axis[1,1].set_title("Projection Phase")
        masksThz.append(proj)
        if not i:
            print("estimated duration task 1: %s" %((np.datetime64("now") - start) * N ** 2))
    start = np.datetime64("now")
    ani = animation.FuncAnimation(fig, heatmap, init_func=init, frames=N*N, repeat=False)
    ani.save("%s.mp4"%title, fps=8)
    masksThz = np.asarray(masksThz)
    coherenceHadamard = []
    coherenceThz = []
    for i in range(N*N):
        for j in range(i):
            coherenceHadamard.append(np.dot(np.conjugate(masks[:,i].T), masks[:,j]))
            coherenceThz.append(np.dot(np.conjugate(masksThz[:,i].T), masksThz[:,j]))
    print(np.max(coherenceHadamard))
    print(np.max(coherenceThz))
    with open("coherence.txt", "w") as f:
        for x, y in zip(coherenceHadamard, coherenceThz):
            f.write("%.1f,%.1f\n"%(np.absolute(x),np.absolute(y)))
        

def analysisFOV():
    w0 = 1
    wavelength = 0.856
    N = 100
    Delta = 0.5
    f = 100
    gb = GaussianBeam(wavelength, w0)
    X, Y = generateFxFy(N, N, Delta, Delta)
    frames = 1000
    if stupidTest: frames = 4
    fig, axis = plt.subplots(2,2)
    fig.suptitle('Beam transformation')
    fig.tight_layout()
    def init():
        sns.heatmap(np.zeros((N,N)), ax=axis[0,0], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[0,1], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[1,0], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[1,1], cbar=False)
    def heatmap(i):
        [[(i.clear(), i.set_axis_off()) for i in ax] for ax in axis]
        z = f + i
        e, p = gb.computeE(X, Y, f=f, z=f)
        e = e * p
        df = DiscreteFresnel(wavelength, Delta, Delta, Delta, Delta, N, N)
        dms = df.computeGD(z)
        EF = np.dot(dms, e.reshape(-1)).reshape((N,N))
        e, p = gb.computeE(X, Y, f=f, z=z)
        EG = e * p
        sns.heatmap(np.absolute(EF), ax=axis[0,0], cbar=False)
        axis[0,0].set_title('Discrete Fresnel Amplitude for i:%d mm'%i)
        sns.heatmap(np.angle(EF), ax=axis[0,1], cbar=False)
        axis[0,1].set_title('Discrete Fresnel Phase for i:%d mm'%i)
        sns.heatmap(np.absolute(EG), ax=axis[1,0], cbar=False)
        axis[1,0].set_title('Gaussian Beam Amplitude for i:%d mm'%i)
        sns.heatmap(np.angle(EG), ax=axis[1,1], cbar=False)
        axis[1,1].set_title('Gaussian Beam Phase for i:%d mm'%i)
        if not i:
            print("Estimate duration task 2: %s" % ((np.datetime64("now") - start) * frames))
    start = np.datetime64("now")
    gaussian = animation.FuncAnimation(fig, heatmap, init_func=init, frames=frames, repeat=False)
    gaussian.save("beam transformation.mp4", fps=4)


def analysisProp():
    w0 = 1
    wavelength = 0.856
    N = 32
    f = 100
    Delta = np.arange(100) * 0.05 + 0.05
    z = np.arange(100) * 10 + 10
    fig, axis=plt.subplots(2,2)
    fig.suptitle("Propagation analysis")
    fig.tight_layout()
    if stupidTest: z, Delta = [10, 100], [0.05, .5]
    def init():
        sns.heatmap(np.zeros((N,N)), ax=axis[0,0], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[0,1], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[1,0], cbar=False)
        sns.heatmap(np.zeros((N,N)), ax=axis[1,1], cbar=False)

    def heatmap(i):
        k = int(i / len(z))
        l = i % len(z)
        [[(i.clear(), i.set_axis_off()) for i in ax] for ax in axis]
        gb = GaussianBeam(wavelength, w0)
        X, Y = generateFxFy(N, N, Delta[l], Delta[l])
        e, p = gb.computeE(X, Y, z=f+10, f=f)
        mask = hadamard(N*N)[0]
        maskThz = (e * p ).reshape(-1) * mask
        df = DiscreteFresnel(wavelength, Delta[l], Delta[l], Delta[l], Delta[l], N, N)
        Dms = df.computeGD(z[k])
        Dsl = df.computeGD(z[k] + 10)
        XS = np.dot(Dms, maskThz)
        XL = np.dot(Dsl, XS)
        sns.heatmap(np.absolute(XS.reshape((N,N))), ax=axis[0,0], cbar=False)
        axis[0,0].set_title("Amplitude at %d mm from mask, with %d micrometer as pixel size"%(z[k], Delta[l] * 1000))
        sns.heatmap(np.angle(XS.reshape((N,N))), ax=axis[0,1], cbar=False)
        axis[0,1].set_title("Phase at %d mm from mask, with %d micrometer as pixel size"%(z[k], Delta[l]*1000))
        sns.heatmap(np.absolute(XL.reshape((N,N))), ax=axis[1,0], cbar=False)
        axis[1,0].set_title("Amplitude at %d mm from scene, with %d micrometer as pixel size"%(z[k], Delta[l]*1000))
        sns.heatmap(np.angle(XL.reshape((N,N))), ax=axis[1,1], cbar=False)
        axis[1,1].set_title("Phase at %d mm from scene, with %d micrometer as pixel size"%(z[k], Delta[l]*1000))
        if not i:
            print("Estimated duration task 3: %s"%((np.datetime64("now") - start) * len(Delta) * len(z)))
    start = np.datetime64("now")
    prop = animation.FuncAnimation(fig, heatmap, init_func=init, frames=len(Delta)*len(z), repeat=False)
    prop.save("propagation analysis.mp4", fps=40)

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
stupidTest = 0
analysisMask()
analysisFOV()
analysisProp()
#if __name__ == "__main__":
    #main()
