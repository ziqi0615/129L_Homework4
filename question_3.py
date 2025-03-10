#!/usr/bin/env python

from numpy import arange
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# b: Lorenz System Simulation
def lorenz_system(time, state, sigma_param, rho_param, beta_param):
    x_coord, y_coord, z_coord = state
    return [
        sigma_param * (y_coord - x_coord),
        x_coord * (rho_param - z_coord) - y_coord,
        x_coord * y_coord - beta_param * z_coord
    ]

time_steps = arange(12, step=1 / 60)
solution_x, solution_y, solution_z = solve_ivp(
    lorenz_system, (time_steps[0], time_steps[-1]), [-20, -20, 20], 
    args=(10, 48, 3), t_eval=time_steps
).y

plt.plot(time_steps, solution_x, label='$x$')
plt.plot(time_steps, solution_y, label='$y$')
plt.plot(time_steps, solution_z, label='$z$')
plt.xlabel('$t$')
plt.legend()
plt.show()

# c: 3D Animation of Lorenz Attractor
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 60)

trajectory_line, = ax.plot([], [], [], lw=1)
time_display = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def update_frame(frame_index):
    trajectory_line.set_data(solution_x[:frame_index], solution_y[:frame_index])
    trajectory_line.set_3d_properties(solution_z[:frame_index])
    time_display.set_text(f'$t={time_steps[frame_index]:.2f}$')
    return trajectory_line, time_display

animation = FuncAnimation(fig, update_frame, frames=len(time_steps), blit=True)
animation.save('lorenz_system.mkv', writer='ffmpeg', fps=60)
