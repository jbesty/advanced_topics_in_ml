import numpy as np
from pyDOE import lhs
from report.ode_solver import solve_ode
import functools


def input_data_initialised(n_ops, power_system):
    time_zeros = np.zeros((n_ops, 1))
    power_zeros = np.zeros((n_ops, power_system['n_buses']))
    delta_initial_zeros = np.zeros((n_ops, power_system['n_buses']))
    omega_initial_zeros = np.zeros((n_ops, power_system['n_buses']))
    delta_results_zeros = np.zeros((n_ops, power_system['n_buses']))
    omega_results_zeros = np.zeros((n_ops, power_system['n_buses']))
    data_type_zeros = np.zeros((n_ops, power_system['n_buses']))

    data_initialised = {'time': time_zeros,
                        'power': power_zeros,
                        'delta_initial': delta_initial_zeros,
                        'omega_initial': omega_initial_zeros,
                        'delta_results': delta_results_zeros,
                        'omega_results': omega_results_zeros,
                        'data_type': data_type_zeros}

    return data_initialised


def sample_power_time_ops(n_ops, t_max, seed, power_system, collocation=True):
    data_ops = input_data_initialised(n_ops=n_ops,
                                      power_system=power_system)
    np.random.seed(seed=seed)
    time_samples = np.random.random((n_ops, 1)) * t_max
    power_samples = power_system['limit_power_lower'] + lhs(n=power_system['n_buses'],
                                                            samples=n_ops) * (
                            power_system['limit_power_upper'] - power_system['limit_power_lower'])

    if not collocation:
        data_type = np.ones((n_ops, power_system['n_buses']))
    else:
        data_type = np.zeros((n_ops, power_system['n_buses']))

    data_ops.update(time=time_samples,
                    power=power_samples,
                    data_type=data_type)
    return data_ops


def evaluate_ops(data_ops, power_system):
    states_initial = np.concatenate([data_ops['delta_initial'],
                                     data_ops['omega_initial']], axis=1)

    t_span = np.concatenate([data_ops['time'] * 0,
                             data_ops['time']], axis=1)

    solver_func = functools.partial(solve_ode,
                                    n_buses=power_system['n_buses'],
                                    lambda_m=power_system['lambda_m'],
                                    lambda_d=power_system['lambda_d'],
                                    lambda_b=power_system['lambda_b'])

    solver_results = map(solver_func, t_span, data_ops['time'],
                         data_ops['power'], states_initial)

    solver_results_list = list(solver_results)
    solver_results_array = np.concatenate(solver_results_list, axis=0)

    data_ops.update(delta_results=solver_results_array[:, 0, :],
                    omega_results=solver_results_array[:, 1, :])
    return data_ops


def create_training_data(n_ops_data, seed_ops_data, n_ops_collocation, seed_ops_collocation, t_max, power_system):
    data_ops_collocation = sample_power_time_ops(n_ops=n_ops_collocation,
                                                 t_max=t_max,
                                                 seed=seed_ops_collocation,
                                                 power_system=power_system,
                                                 collocation=True)
    data_ops_data = sample_power_time_ops(n_ops=n_ops_data,
                                          t_max=t_max,
                                          seed=seed_ops_data,
                                          power_system=power_system,
                                          collocation=False)
    data_ops_data = evaluate_ops(data_ops_data, power_system)

    data_ops_combined = {}
    for key in data_ops_data:
        data_ops_combined[key] = np.concatenate([data_ops_data[key],
                                                 data_ops_collocation[key]], axis=0)

    return data_ops_combined


def create_test_data(n_ops_data, seed_ops_data,t_max, power_system):
    data_ops_data = sample_power_time_ops(n_ops=n_ops_data,
                                          t_max=t_max,
                                          seed=seed_ops_data,
                                          power_system=power_system,
                                          collocation=False)
    data_ops_data = evaluate_ops(data_ops_data, power_system)

    return data_ops_data



if __name__ == "__main__":
    from report.power_system_handling import load_power_system

    power_system = load_power_system(n_buses=1)
    test = create_training_data(n_ops_data=20,
                                seed_ops_data=2,
                                n_ops_collocation=100,
                                seed_ops_collocation=3,
                                t_max=5,
                                power_system=power_system)
