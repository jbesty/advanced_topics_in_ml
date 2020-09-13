import numpy as np
from scipy import integrate


def ode_right_hand_side(t, state_variable, n_buses, lambda_m, lambda_d, lambda_b, power):
    """
    system of first order ordinary differential equations
    :param t: variable if ode depends on t
    :param state_variable: state vector consisting of delta_i and omega_i for i in 1:n_buses
    :param n_buses: number of buses
    :param lambda_m: inertia at each bus
    :param lambda_d: damping coefficient at each bus
    :param lambda_b: bus susceptance matrix
    :param power: power injection or retrieval at each bus
    :return: updated state variable
    """
    # split the state variable into delta and omega
    state_delta = state_variable[:n_buses].reshape((-1, 1))
    state_omega = state_variable[n_buses:].reshape((-1, 1))

    # computing the non-linear term in the swing equation sum_j (B_ij sin(delta_i - delta_j))
    delta_i = np.repeat(state_delta, repeats=n_buses, axis=1)
    if n_buses == 1:
        delta_j = np.zeros(shape=delta_i.shape)
    else:
        delta_j = np.repeat(state_delta.reshape((1, -1)), repeats=n_buses, axis=0)

    delta_ij = np.sin(delta_i - delta_j)
    connectivity_vector = np.sum(np.multiply(lambda_b, delta_ij), axis=1).reshape((-1, 1))

    # update states
    state_delta_new = np.zeros(shape=state_delta.shape)
    state_omega_new = np.zeros(shape=state_omega.shape)

    for bus in range(n_buses):
        if lambda_m[bus] > 0:
            state_delta_new[bus] = state_omega[bus]
            state_omega_new[bus] = 1 / lambda_m[bus] * (
                    power[bus] - lambda_d[bus] * state_omega[bus] - connectivity_vector[bus])
        elif lambda_d[bus] > 0:
            state_delta_new[bus] = 1 / lambda_d[bus] * (power[bus] - connectivity_vector[bus])
            state_omega_new[bus] = 0
        else:
            state_delta_new[bus] = 0
            state_omega_new[bus] = 0

    return np.concatenate([state_delta_new[:, 0],
                           state_omega_new[:, 0]],
                          axis=0)


def evaluate_algebraic_equations(state_evolution, n_buses, lambda_m, lambda_d, lambda_b, power):
    """
    evaluate all states (here omega_i for buses with lambda_m == 0) described by algebraic equations
    :param state_evolution: state variables over time of shape [n_timesteps, states, n_buses]
        where states represents delta and omega
    :param n_buses: number of buses
    :param lambda_m: inertia at each bus
    :param lambda_d: damping coefficient at each bus
    :param lambda_b: bus susceptance matrix
    :param power: power injection or retrieval at each bus
    :return: updated state_evolution
    """

    # computing the non-linear term in the swing equation Sum_j [B_ij sin(delta_i - delta_j)]
    lambda_b = lambda_b.reshape(1, n_buses, n_buses)
    delta_i = np.repeat(state_evolution[:, 0, :].reshape((-1, n_buses, 1)),
                        repeats=n_buses,
                        axis=2)

    if n_buses == 1:
        delta_j = delta_i * 0
    else:
        delta_j = np.repeat(state_evolution[:, 0, :].reshape((-1, 1, n_buses)),
                            repeats=n_buses,
                            axis=1)

    connectivity_matrix = lambda_b * np.sin(delta_i - delta_j)
    connectivity_vector = np.sum(connectivity_matrix, axis=2)

    # update states for all time steps at once
    for bus in range(n_buses):
        if lambda_m[bus] > 0:
            pass
        elif lambda_d[bus] > 0:
            state_evolution[:, 1, bus] = 1 / lambda_d[bus] * (power[bus] - connectivity_vector[:, bus])
        else:
            # TODO: evaluate algebraic equations for buses with non-frequency dependent load
            pass

    return state_evolution


def solve_ode(t_span,
              t_eval,
              power,
              states_initial,
              n_buses,
              lambda_m,
              lambda_d,
              lambda_b):
    ode_solution = integrate.solve_ivp(ode_right_hand_side,
                                       t_span=t_span,
                                       y0=states_initial,
                                       args=[n_buses, lambda_m, lambda_d, lambda_b, power],
                                       t_eval=t_eval,
                                       rtol=1e-5)

    # array in the form of [states x evaluation points] -> [evaluation points x states x buses]
    state_results_incomplete = np.transpose(ode_solution.y).reshape((-1, 2, n_buses))

    state_results_complete = evaluate_algebraic_equations(state_results_incomplete, n_buses, lambda_m, lambda_d,
                                                          lambda_b, power)

    return state_results_complete


if __name__ == '__main__':
    from report.power_system_handling import load_power_system

    power_system = load_power_system(n_buses=1)
    t_eval = np.linspace(0, 10, 1000)

    # t_eval = np.array([0.0, 1.0, 5.0, 10.0])
    t_span = np.array([0.0, max(t_eval)])
    # states_initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states_initial = np.array([0.0, 0.0])
    power = np.array([0.3222])
    results = solve_ode(t_span=t_span,
                        t_eval=t_eval,
                        power=power,
                        states_initial=states_initial,
                        n_buses=power_system['n_buses'],
                        lambda_m=power_system['lambda_m'],
                        lambda_d=power_system['lambda_d'],
                        lambda_b=power_system['lambda_b'])
