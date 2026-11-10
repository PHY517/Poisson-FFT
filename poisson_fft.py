import numpy as np
import matplotlib.pyplot as plt

# --- 1. Real-space grid (periodic box, no time involved) ---
N = 128
L = 10.0
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")
dx = x[1] - x[0]

# --- 2. Charge density: a +/- pair of Gaussian charges ---
def gauss(x0, y0, s=0.5):
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*s**2))

rho = gauss(3.5, 5.0) - gauss(6.5, 5.0)
rho -= rho.mean()            # enforce neutrality (k = 0 term vanishes)

# --- 3. Poisson equation  -laplacian(phi) = rho  in Fourier space ---
rho_k = np.fft.fft2(rho)

k = 2*np.pi*np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(k, k, indexing="ij")
K2 = KX**2 + KY**2
K2[0, 0] = 1.0               # avoid division by zero

phi_k = rho_k / K2           # phi(k) = rho(k) / k^2
phi_k[0, 0] = 0.0            # choose the zero of the potential
phi = np.real(np.fft.ifft2(phi_k))

# --- 4. Check: apply the Laplacian in Fourier space ---
lap_phi = np.real(np.fft.ifft2(-K2 * np.fft.fft2(phi)))
print("max |-lap(phi) - rho| =", np.max(np.abs(-lap_phi - rho)))

# --- 5. Plot ---
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
im0 = ax[0].imshow(rho.T, origin="lower",
                   extent=[0, L, 0, L], cmap="RdBu_r")
ax[0].set_title(r"Charge density $\rho(x,y)$")
im1 = ax[1].imshow(phi.T, origin="lower",
                   extent=[0, L, 0, L], cmap="viridis")
ax[1].set_title(r"Potential $\phi(x,y)$ from FFT")
ax[2].imshow(np.log1p(np.abs(np.fft.fftshift(rho_k))),
             cmap="magma")
ax[2].set_title(r"$|\rho(k_x,k_y)|$ (log scale)")
for a in ax[:2]:
    a.set_xlabel("x"); a.set_ylabel("y")
ax[2].set_xlabel(r"$k_x$ index"); ax[2].set_ylabel(r"$k_y$ index")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
fig.colorbar(im1, ax=ax[1], fraction=0.046)
plt.tight_layout()
plt.show()



