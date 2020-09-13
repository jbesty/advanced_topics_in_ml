# structure adopted from https://github.com/mardani72/Hyper-Parameter_optimization

import skopt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from skopt import plots
from report.create_training_data import create_training_data, create_test_data
from report.power_system_handling import load_power_system
from report.create_model import PinnModel

np.random.seed(12345)
sampling_seeds = np.random.randint(low=1, high=100000, size=1000)
power_system = load_power_system(n_buses=1)

n_ops_data = 100
n_ops_test = 200
n_ops_collocation = 900
t_max = 5
data_ratio_max = (n_ops_collocation + n_ops_data) / n_ops_data

# ------------------------------------------------------
# Data preparation
training_data = create_training_data(n_ops_data=n_ops_data,
                                     seed_ops_data=sampling_seeds[0],
                                     n_ops_collocation=n_ops_collocation,
                                     seed_ops_collocation=sampling_seeds[1],
                                     t_max=t_max,
                                     power_system=power_system)

X_train = [training_data['time'],
           training_data['power'],
           training_data['data_type']]

y_train = [training_data['delta_results'],
           training_data['omega_results'],
           np.zeros(training_data['delta_results'].shape)]

test_data = create_test_data(n_ops_data=n_ops_test,
                             seed_ops_data=sampling_seeds[2],
                             t_max=t_max,
                             power_system=power_system)

X_test = [test_data['time'],
          test_data['power'],
          test_data['data_type']]

y_test = [test_data['delta_results'],
          test_data['omega_results'],
          np.zeros(test_data['delta_results'].shape)]
# ------------------------------------------------------

# ------------------------------------------------------
# Hyper-parameter definition
dim_initial_learning_rate = skopt.space.Real(low=1e-4, high=1e-1, prior='log-uniform', name='initial_learning_rate')
dim_decay_learning_rate = skopt.space.Real(low=0.1, high=1, prior='uniform', name='decay_learning_rate')
dim_decay_steps = skopt.space.Integer(low=5, high=100, name='decay_steps', prior='uniform')
dim_data_ratio = skopt.space.Real(low=0.1, high=data_ratio_max*10, name='data_ratio', prior='log-uniform')

dimensions = [dim_initial_learning_rate,
              dim_decay_learning_rate,
              dim_decay_steps,
              dim_data_ratio]

default_parameters = [1e-1, 0.95, 10, 1]

# define global variable to store accuracy
best_combined_mse = 10000.0


@skopt.utils.use_named_args(dimensions=dimensions)
def fitness(initial_learning_rate,
            decay_learning_rate,
            decay_steps,
            data_ratio):
    # Create the neural network with these hyper-parameters.
    model = PinnModel(num_dense_layers=3,
                      num_dense_nodes=50,
                      initial_learning_rate=initial_learning_rate,
                      decay_learning_rate=decay_learning_rate,
                      decay_steps=decay_steps,
                      t_max_normaliser=t_max,
                      data_ratio=data_ratio,
                      power_system=power_system)

    log_dir = f".\\logs\\{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    callback_log = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=10,
                        batch_size=25,
                        validation_data=(X_test, y_test),
                        callbacks=[callback_log],
                        verbose=0)

    delta_mse = history.history['val_output_1_loss'][-1]
    omega_mse = history.history['val_output_2_loss'][-1]
    physics_mse = history.history['val_output_3_loss'][-1]

    print('initial_learning rate: {0:.1e}'.format(initial_learning_rate))
    print('decay_learning_rate:', decay_learning_rate)
    print('decay_steps:', decay_steps)
    print('data_ratio:', data_ratio)
    print()
    print("Delta MSE: {0:.4%}".format(delta_mse))
    print("Omega MSE: {0:.4%}".format(omega_mse))
    print("Physics MSE: {0:.4%}".format(physics_mse))
    print()

    global best_combined_mse

    # choose between objectives
    objective_value = np.sqrt(np.mean([delta_mse ** 2, omega_mse ** 2, physics_mse ** 2]))
    # objective_value = delta_mse

    tf.keras.backend.clear_session()

    return objective_value

fitness(x=default_parameters)

search_result = skopt.gp_minimize(func=fitness,
                                  dimensions=dimensions,
                                  acq_func='gp_hedge',  # Expected Improvement.
                                  n_calls=50,
                                  x0=default_parameters)


dim_names = ['Initial learning rate', 'Decay learning rate',
             'Decay steps', 'Data ratio']

opt_par = search_result.x
model = PinnModel(num_dense_layers=3,
                  num_dense_nodes=50,
                  initial_learning_rate=opt_par[0],
                  decay_learning_rate=opt_par[1],
                  decay_steps=opt_par[2],
                  t_max_normaliser=t_max,
                  data_ratio=opt_par[3],
                  power_system=power_system)

log_dir = f".\\logs\\{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

callback_log = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=False)

history = model.fit(x=X_train,
                    y=y_train,
                    epochs=500,
                    batch_size=100,
                    validation_data=(X_test, y_test),
                    callbacks=[callback_log],
                    verbose=2)

print(f'Final MSE for delta {history.history["val_output_1_loss"][-1]*100} %')
print(f'Final MSE for omega {history.history["val_output_2_loss"][-1]*100} %')
print(f'Final MSE for f {history.history["val_output_3_loss"][-1]*100} %')

skopt.plots.plot_convergence(search_result)
plt.yscale('log')
plt.show()

fig, axs = skopt.plots.plot_objective(result=search_result, dimensions=dim_names)
plt.show()