import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.stats import linregress

# Constants
G = 10
N = 100
dt = 0.01
softening = 1e-5
box_size = 40
t_max = 3
frames_max = int(t_max / dt)
r_param = 2

# Initialize particles
positions = np.random.rand(N, 2) * (2 * box_size - 1) - (2 * box_size - 1) / 2
velocities = np.random.randn(N, 2) * 0.75
masses = np.ones(N)

# Compute gravitational forces
def compute_forces(positions, masses):
    forces = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r = positions[j] - positions[i]
            dist = np.linalg.norm(r) + softening
            force = G * masses[i] * masses[j] / dist**(r_param + 1) * r
            forces[i] += force / masses[i]
            forces[j] -= force / masses[j]
    return forces

# Plot mass distribution
def show_mass_histogram(masses):
    log_masses = np.log10(masses)

    # Bin the log-masses
    counts, bins = np.histogram(log_masses, bins='auto')
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Filter out zero-count bins
    nonzero = counts > 0
    x = bin_centers[nonzero]
    y = np.log10(counts[nonzero])

    # Fit linear regression to log-log histogram
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_fit = slope * x + intercept

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(x, y, width=np.diff(bins)[0], align='center', label='Data')
    plt.plot(x, y_fit, linewidth=2, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}', color='crimson')
    
    plt.xlabel('log(M)')
    plt.ylabel('log(N)')
    plt.title('Log-Log Mass Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Linear regression: slope = {slope:.3f}, intercept = {intercept:.3f}, R2 = {r_value**2:.3f}")

# Handle collisions and merge particles
def handle_collisions(positions, velocities, masses):
    merged = set()
    new_positions = []
    new_velocities = []
    new_masses = []
    for i in range(len(positions)):
        if i in merged:
            continue
        group = [i]
        for j in range(i + 1, len(positions)):
            if j in merged:
                continue
            distance = np.linalg.norm(positions[j] - positions[i])
            radius_i = masses[i] ** (1/3)
            radius_j = masses[j] ** (1/3)
            dynamic_collision_distance = (radius_i + radius_j) * 0.25
            if distance < dynamic_collision_distance:
                group.append(j)
        if len(group) > 1:
            merged.update(group)
            total_mass = sum(masses[k] for k in group)
            merged_pos = sum(masses[k] * positions[k] for k in group) / total_mass
            merged_vel = sum(masses[k] * velocities[k] for k in group) / total_mass
            new_positions.append(merged_pos)
            new_velocities.append(merged_vel)
            new_masses.append(total_mass)
        else:
            new_positions.append(positions[i])
            new_velocities.append(velocities[i])
            new_masses.append(masses[i])
    return np.array(new_positions), np.array(new_velocities), np.array(new_masses)

# Update function for animation
def update(frame):
    global positions, velocities, masses, circles

    forces = compute_forces(positions, masses)
    velocities += 0.5 * forces * dt
    positions += velocities * dt
    forces_new = compute_forces(positions, masses)
    velocities += 0.5 * forces_new * dt
    positions, velocities, masses = handle_collisions(positions, velocities, masses)

    # Toroidal (wrapping) boundaries
    positions = (positions + box_size) % (2 * box_size) - box_size

    # Update scatter plot
    scat.set_offsets(positions)
    scat.set_array(np.log10(masses))
    scat.set_sizes(masses * 10)
    scat.set_clim(np.log10(1), np.log10(N))

    # Update circle outlines
    for i, circle in enumerate(circles):
        if i < len(positions):
            radius = masses[i] ** (1/3)
            circle.set_center(positions[i])
            circle.set_radius(radius)
            circle.set_visible(True)
        else:
            circle.set_visible(False)

    time_text.set_text(f'Time: {frame * dt:.2f}')
    return [scat, time_text] + circles

# Set up plot
fig, ax = plt.subplots()
ax.set_xlim(-box_size, box_size)
ax.set_ylim(-box_size, box_size)
ax.set_facecolor('grey')

scat = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    c=np.log10(masses),
    s=masses * 10,
    cmap='inferno',
    vmin=np.log10(1),
    vmax=np.log10(N)
)

# Create circles for mass-based radii
circles = []
for i in range(len(positions)):
    radius = masses[i] ** (1/3)
    circle = Circle(positions[i], radius, fill=False, edgecolor='white', linewidth=0.5, alpha=0.3)
    ax.add_patch(circle)
    circles.append(circle)

time_text = ax.text(-box_size + 2, box_size - 5, '', color='white', fontsize=10)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=frames_max, interval=5, blit=False, repeat=False)
cbar = plt.colorbar(scat, ax=ax, label='log(M)')
plt.show()

print('Simulation completed')
show_mass_histogram(masses)