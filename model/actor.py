import tensorflow as tf
import numpy as np

initializer = tf.variance_scaling_initializer()
activation = tf.nn.elu
learning_rate = 1e-3

class Actor:

    def __init__(self, n_state, n_action, n_units, n_layers, scope):
        self.scope = scope
        self.n_state = n_state
        self.n_action = n_action
        self.n_units = n_units
        self.n_layers = n_layers

        self._build_network()

    def set_session(self, sess : tf.Session):
        self.session = sess

    def _build_network(self):

        with tf.variable_scope(self.scope) as scope:
            # state input of the actor
            self._state = tf.placeholder(dtype=tf.float32, shape=(None, self.n_state))
            self.q_gradient = tf.placeholder("float", [None, self.n_action])
            # first layer
            hidden = tf.layers.dense(inputs=self._state,
                                     units=self.n_units,
                                     activation=activation,
                                     kernel_initializer=initializer)
            # the rest of the layers
            for _ in range(self.n_layers - 1):
                hidden = tf.layers.dense(inputs=hidden,
                                         units=self.n_units,
                                         activation=activation,
                                         kernel_initializer=initializer)

            self._action = tf.layers.dense(inputs=hidden,
                                          units=self.n_action,
                                          activation=tf.nn.tanh,
                                          kernel_initializer=initializer)

            self.parameters_gradients = tf.gradients(self._action, self.get_variables(), -self.q_gradient)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.parameters_gradients, self.get_variables()))

    def action(self, state):
        state = np.reshape(state, newshape=(1, 2))
        return self.session.run(self._action, feed_dict={self._state: state})

    def actions(self, state_batch):
        return self.session.run(self._action, feed_dict={self._state: state_batch})

    def update(self, state_batch, gradient_batch):
        self.session.run(self.optimizer, feed_dict={self._state: state_batch, self.q_gradient: gradient_batch})
        return True

    def get_variables(self):
        return tf.trainable_variables(scope=self.scope)
