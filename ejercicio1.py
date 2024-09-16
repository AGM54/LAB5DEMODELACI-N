import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

M, N = 10, 10  
r, s = 3, 3  
T = 100  
K = 0.1  
neigh = 8  


def create_mask(M, N, r, s):
    mask = np.ones((M, N))
    mask[:r, :s] = 0  
    return mask


def initialize_distribution(M, N):
    u0 = np.random.rand(M, N)  # Distribución aleatoria inicial
    return u0 / np.sum(u0)  # Normalizado

def diffusion_step(u, mask, K):
    u_next = np.copy(u)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if mask[i, j] == 1:  # Solo actualizar si está dentro de la región
                neighbors_sum = (
                    u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] +
                    u[i+1, j+1] + u[i+1, j-1] + u[i-1, j+1] + u[i-1, j-1]
                )
                u_next[i, j] = (1 - K) * u[i, j] + (K / 8) * neighbors_sum
    return u_next


def simulate_diffusion(M, N, r, s, T, K):
    mask = create_mask(M, N, r, s)  
    u = initialize_distribution(M, N)  
    history = [u]  

    for t in range(T):
        u = diffusion_step(u, mask, K)
        history.append(u)

    return history


def plot_diffusion(history):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        ax.set_title(f'Time: {frame}')
        im = ax.imshow(history[frame], cmap='viridis', interpolation='none')
        return im,
    
    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False)
    plt.show()


history = simulate_diffusion(M, N, r, s, T, K)
plot_diffusion(history)
