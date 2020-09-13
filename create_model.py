import tensorflow as tf
import numpy as np


class DenseCoreNetwork(tf.keras.models.Model):

    def __init__(self, neurons_in_hidden_layer, t_max_normaliser, power_system):

        super(DenseCoreNetwork, self).__init__()

        self.n_buses = power_system['n_buses']
        self.neurons_in_hidden_layers = neurons_in_hidden_layer
        self.t_max_normaliser = t_max_normaliser
        self.power_system = power_system
        self.power_no_variation = np.equal(power_system['limit_power_lower'],
                                           power_system['limit_power_upper'])

        self.hidden_layers = []
        for n_neurons in self.neurons_in_hidden_layers:
            self.hidden_layers.append(tf.keras.layers.Dense(units=n_neurons,
                                                            activation=tf.keras.activations.tanh,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                                            bias_initializer=tf.keras.initializers.zeros,
                                                            ))

        self.dense_output_layer = tf.keras.layers.Dense(units=self.n_buses,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                                        bias_initializer=tf.keras.initializers.zeros)

    def call(self, inputs, training=None, mask=None):

        time_input, power_input = inputs

        time_normalised = 2.0 * time_input / self.t_max_normaliser - 1.0

        # ensure that in case the lower and upper bounds are fixed, the denominator is not 0 and the normalised term
        # evaluates to 0.
        power_normalised_list = []
        for bus, no_variation in enumerate(self.power_no_variation):
            power_input_bus = power_input[:, bus:bus + 1]
            if no_variation:
                power_normalised_list.append(power_input_bus * 0.0)
            else:
                power_normalised_list.append(2.0 * (power_input_bus - self.power_system['limit_power_lower'][bus]) / (
                        self.power_system['limit_power_upper'][bus] - self.power_system['limit_power_lower'][
                    bus]) - 1.0)

        power_normalised = tf.concat(power_normalised_list, axis=1)

        normalised_inputs = tf.concat([time_normalised, power_normalised], axis=1)

        hidden_layer_input = normalised_inputs

        for layer in self.hidden_layers:
            hidden_layer_input = layer(hidden_layer_input)

        network_output = self.dense_output_layer(hidden_layer_input)

        return network_output


class PinnLayer(tf.keras.layers.Layer):

    def __init__(self, neurons_in_hidden_layer, t_max_normaliser, power_system):
        super(PinnLayer, self).__init__()

        self.n_buses = power_system['n_buses']

        self.lambda_m = tf.Variable(power_system['lambda_m'].reshape((1, self.n_buses)),
                                    trainable=False,
                                    name='lambda_m',
                                    dtype=tf.float32)

        self.lambda_d = tf.Variable(power_system['lambda_d'].reshape((1, self.n_buses)),
                                    trainable=False,
                                    name='lambda_d',
                                    dtype=tf.float32)

        self.lambda_b = tf.Variable(power_system['lambda_b'].reshape((self.n_buses, self.n_buses)),
                                    trainable=False,
                                    name='lambda_b',
                                    dtype=tf.float32)

        self.DenseLayers = DenseCoreNetwork(neurons_in_hidden_layer=neurons_in_hidden_layer,
                                            t_max_normaliser=t_max_normaliser,
                                            power_system=power_system)

    def calculate_time_derivatives(self, inputs):
        time_input, _ = inputs

        list_network_output = []
        list_network_output_t = []
        list_network_output_tt = []

        for bus in range(self.n_buses):
            with tf.GradientTape(watch_accessed_variables=False,
                                 persistent=False) as grad_tt:
                grad_tt.watch(time_input)
                with tf.GradientTape(watch_accessed_variables=False,
                                     persistent=False) as grad_t:
                    grad_t.watch(time_input)
                    network_output_single = self.DenseLayers(inputs)[:, bus:bus + 1]

                    network_output_t_single = grad_t.gradient(network_output_single,
                                                              time_input,
                                                              unconnected_gradients='zero')
                network_output_tt_single = grad_tt.gradient(network_output_t_single,
                                                            time_input,
                                                            unconnected_gradients='zero')

            list_network_output.append(network_output_single)
            list_network_output_t.append(network_output_t_single)
            list_network_output_tt.append(network_output_tt_single)

        network_output = tf.concat(list_network_output, axis=1)
        network_output_t = tf.concat(list_network_output_t, axis=1)
        network_output_tt = tf.concat(list_network_output_tt, axis=1)

        return network_output, network_output_t, network_output_tt

    def call(self, inputs, **kwargs):
        _, power_input = inputs

        network_output, network_output_t, network_output_tt = self.calculate_time_derivatives(inputs=inputs)

        delta_i = tf.repeat(input=tf.reshape(network_output, [-1, self.n_buses, 1]),
                            repeats=self.n_buses,
                            axis=2)

        if self.n_buses == 1:
            delta_j = delta_i * 0
        else:
            delta_j = tf.repeat(input=tf.reshape(network_output, [-1, 1, self.n_buses]),
                                repeats=self.n_buses,
                                axis=1)

        connectivity_matrix = self.lambda_b * tf.math.sin(delta_i - delta_j)
        connectivity_vector = tf.reduce_sum(connectivity_matrix, axis=2)

        network_output_physics = (self.lambda_m * network_output_tt +
                                  self.lambda_d * network_output_t +
                                  connectivity_vector - power_input)

        return network_output, network_output_t, network_output_physics


class PinnModel(tf.keras.models.Model):

    def __init__(self,
                 num_dense_layers,
                 num_dense_nodes,
                 initial_learning_rate,
                 decay_learning_rate,
                 decay_steps,
                 t_max_normaliser,
                 data_ratio,
                 power_system,
                 seed=2345):
        super(PinnModel, self).__init__()
        tf.random.set_seed(seed)
        neurons_in_hidden_layer = [num_dense_nodes] * int(num_dense_layers)

        self.n_buses = power_system['n_buses']
        self.PinnLayer = PinnLayer(neurons_in_hidden_layer=neurons_in_hidden_layer,
                                   t_max_normaliser=t_max_normaliser,
                                   power_system=power_system)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                          decay_steps=decay_steps,
                                                                          decay_rate=decay_learning_rate,
                                                                          staircase=True)
        self.loss_weights = [data_ratio,
                             data_ratio,
                             1]

        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_schedule),
                     loss=tf.keras.losses.mean_squared_error,
                     loss_weights=self.loss_weights)

        self.build(input_shape=[(None, 1), (None, self.n_buses), (None, self.n_buses)])

    def call(self, inputs, training=None, mask=None):
        x_time, x_power, x_type = inputs

        network_output, network_output_t, network_output_physics = self.PinnLayer([x_time, x_power])

        loss_network_output = tf.multiply(network_output, x_type)
        loss_network_output_t = tf.multiply(network_output_t, x_type)

        loss_network_output_physics = network_output_physics

        return loss_network_output, loss_network_output_t, loss_network_output_physics
