import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def compute_nj(N, t):
    j0 = 1  # site perturbé (1-indexé)

    # Hamiltonien tight-binding complet : saut i <-> i+1 pour tout i
    h = np.zeros((N, N))
    for i in range(N - 1):
        h[i, i + 1] = h[i + 1, i] = 1.0
    # Diagonalisation du hamiltonien
    epsilon, phi = np.linalg.eigh(h)  # epsilon: énergies, phi: fonctions d'ondes (N, N) avec phi[n, j] = φₙ(j)

    particle_modes = np.where(epsilon > 0)[0]
    hole_modes = np.where(epsilon <= 0)[0]
    # Terme dynamique
    weights = phi[particle_modes, j0 - 1]  # phi_n(j0)
    # norme² de c†_{j0}|FS⟩ : site j0 peut être partiellement occupé dans |FS⟩
    A = (weights**2).sum()
    propagator = phi[particle_modes, :] * weights[:, None] \
               * np.exp(-1j * epsilon[particle_modes, None] * t)
    term1 = np.abs(propagator.sum(axis=0))**2 / A

    # Terme GS : occupation de fond
    term2 = (phi[hole_modes, :]**2).sum(axis=0)  # sum_{n<=kF} |phi_n(j)|^2

    return term1 + term2

"""
sites = np.arange(1, N + 1)
times = np.arange(0, 10, 0.05)
total_particles = np.array([compute_nj(t).sum() for t in times])
print(f"Nombre total de particules a t=0 : {total_particles[0]:.2f} (devrait être {NF+1})")

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7))
nj0 = compute_nj(0)
line, = ax.plot(sites, nj0, lw=2)
ax.set_ylim(0, 1.2 * nj0.max())
ax.set_xlabel('site $j$')
ax.set_ylabel(r'$\langle n_j(t)\rangle$')
ax.axhline(0.5, color='red', ls='--', alpha=0.4, label='demi-remplissage')
ax.axvline(j0, color='green', ls=':', alpha=0.5, label=f'perturbation j={j0}')
title = ax.set_title('t = 0.00')
ax.legend()

particle_line, = ax2.plot(times, total_particles, lw=2, color='tab:orange')
ax2.set_ylim(0, 1.1 * total_particles.max())
ax2.set_xlim(times[0], times[-1])
ax2.set_xlabel('temps $t$')
ax2.set_ylabel('nombre total de particules')
ax2.set_title('Nombre total de particules en fonction du temps')

def update(frame):
    nj = compute_nj(times[frame])
    line.set_ydata(nj)
    title.set_text(f't = {times[frame]:.2f}')
    particle_line.set_data(times[:frame + 1], total_particles[:frame + 1])
    return line, particle_line

ani = animation.FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
plt.tight_layout()
plt.show()
"""