import numpy as np
import pickle
import os


def create_power_system():

    n_buses = 1

    lambda_m = np.array([0.4]).reshape((1, n_buses))
    lambda_d = np.array([0.15]).reshape((1, n_buses))
    lambda_b = np.array([0.2]).reshape((n_buses, n_buses))

    limit_power_lower = np.array([0.0]).reshape((1, n_buses))
    limit_power_upper = np.array([0.4]).reshape((1, n_buses))

    bus_with_inertia = lambda_m > 0
    bus_with_damping = lambda_d > 0
    lines_connected = lambda_b > 0

    system_parameters = {'n_buses': n_buses,
                         'lambda_m': lambda_m,
                         'lambda_d': lambda_d,
                         'lambda_b': lambda_b,
                         'limit_power_lower': limit_power_lower,
                         'limit_power_upper': limit_power_upper,
                         'bus_with_inertia': bus_with_inertia,
                         'bus_with_damping': bus_with_damping,
                         'lines_connected': lines_connected}

    file_path = os.path.join('C:/Users/Jochen Stiasny/PyCharmProjects/advanced_topics_in_ml',
                             'system_1_bus.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(system_parameters, f)


def load_power_system(n_buses):
    file_path = os.path.join('C:/Users/Jochen Stiasny/PyCharmProjects/advanced_topics_in_ml',
                             f'system_{n_buses}_bus.pickle')

    with open(file_path, "rb") as f:
        power_system = pickle.load(f)

    return power_system


if __name__ == '__main__':
    create_power_system()

