#!/usr/bin/env python

from numpy import linspace, unique, round, flatnonzero, mean, array, roll, nan
from matplotlib import pyplot as plt

# a: Stability of Fixed Points
for growth_rate in [1, 2, 3, 4]:
    print(f'{growth_rate=}')
    for fixed_point in [0, (growth_rate - 1) / growth_rate]:
        print(f'Fixed point: {fixed_point}')
        if growth_rate * (1 - 2 * fixed_point) < 0:
            print('Stable')
        else:
            print('Unstable')

# b: Convergence of Logistic Map
max_iterations = 1000
convergence_threshold = 1e-6
initial_value = 0.2
for growth_rate in [2, 3, 3.5, 3.8, 4.0]:
    print(f'{growth_rate=}')
    state = last_state = initial_value
    converged = False
    for iteration in range(max_iterations):
        state = growth_rate * state * (1 - state)
        if abs(state - last_state) < convergence_threshold:
            print(f'Converged to {state} in {iteration} iterations')
            converged = True
            break
        last_state = state
    if not converged:
        print(f'Did not converge after {max_iterations} iterations')

# c: Logistic Map Time Evolution
fig, axes = plt.subplots(2, 2)
for idx, growth_rate in enumerate([1, 2, 3, 4]):
    ax = axes[idx // 2, idx % 2]
    ax.set_title(f'$r={growth_rate}$')
    for initial_value in [0.1, 0.3, 0.5]:
        state = initial_value
        state_trajectory = [initial_value]
        for _ in range(500):
            state = growth_rate * state * (1 - state)
            state_trajectory.append(state)
        ax.scatter(range(len(state_trajectory)), state_trajectory, label=f'$x_0={initial_value}$', s=2)
    ax.legend()
    ax.set_xlabel('$n$')
    ax.set_ylabel('$x_n$')
plt.show()

# d: Period Doubling Bifurcation
growth_rate_values = linspace(0.01, 4, 1000)
period_counts = []
for growth_rate in growth_rate_values:
    state = 0.2
    trajectory_samples = []
    for iteration in range(500):
        if iteration > 450:
            trajectory_samples.append(round(state, decimals=6))
        state = growth_rate * state * (1 - state)
    period_counts.append(len(unique(trajectory_samples)))
period_counts = array(period_counts)

def get_period_positions(period_length):
    condition_array = period_counts == period_length
    condition_array[condition_array & ~roll(condition_array, 1) & ~roll(condition_array, -1)] = False
    return flatnonzero(condition_array)

def find_bifurcation_point(from_period, to_period):
    idx1 = get_period_positions(from_period)[-1]
    idx2 = get_period_positions(to_period)[0]
    return (growth_rate_values[idx1] + growth_rate_values[idx2]) / 2

print(f'Period change from 1 to 2 at r={find_bifurcation_point(1, 2):.3f}')
print(f'Period change from 2 to 4 at r={find_bifurcation_point(2, 4):.3f}')
print(f'Period change from 4 to 8 at r={find_bifurcation_point(4, 8):.3f}')
print(f'Chaos starts at r={growth_rate_values[get_period_positions(8)[-1] + 1]:.3f}')
print(f'Period is 3 at r={mean(growth_rate_values[get_period_positions(3)]):.3f}')

# e: Generalized Logistic Map with γ
gamma_values = linspace(0.5, 1.5, 100)
bifurcation_thresholds = []
for gamma in gamma_values:
    bifurcation_detected = False
    appended = False
    for growth_rate in linspace(1.02, 6, 250):
        state = 0.2
        trajectory_samples = []
        for iteration in range(500):
            if iteration > 450:
                trajectory_samples.append(round(state, decimals=6))
            state = growth_rate * state * (1 - state ** gamma)
        if len(unique(trajectory_samples)) == 2:
            if bifurcation_detected:
                bifurcation_thresholds.append(growth_rate)
                appended = True
                break
            bifurcation_detected = True
        else:
            bifurcation_detected = False
    if not appended:
        bifurcation_thresholds.append(nan)

plt.plot(gamma_values, bifurcation_thresholds)
plt.xlabel(r'$\gamma$')
plt.ylabel('Smallest $r$ with bifurcation')
plt.show()
