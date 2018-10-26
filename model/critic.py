import tensorflow as tf
import numpy as np

initializer = tf.variance_scaling_initializer()
activation = tf.nn.elu
learning_rate = 1e-3

class Critic:

    def __init__(self, n_state, n_action, n_units, n_layers, scope):
        # initialize member variables
        self.scope = scope
        self.n_state = n_state
        self.n_action = n_action
        self.n_units = n_units
        self.n_layers = n_layers
        # build network
        self._build_network()

    def set_session(self, sess : tf.Session):
        self.session = sess

    def _build_network(self):
        # set proper variable scope
        with tf.variable_scope(self.scope):
            # entry point
            self._state = tf.placeholder(dtype=tf.float32, shape=(None, self.n_state))
            self._action = tf.placeholder(dtype=tf.float32, shape=(None, self.n_action))
            self._y = tf.placeholder(dtype=tf.float32, shape=(None, self.n_action))

            with tf.variable_scope("state"):
                hidden_state = tf.layers.dense(inputs=self._state, units=self.n_units, activation=activation, kernel_initializer=initializer)
                for _ in range(self.n_layers - 1):
                    hidden_state = tf.layers.dense(inputs=hidden_state, units=self.n_units, activation=activation, kernel_initializer=initializer)

            with tf.variable_scope("action"):
                hidden_action = tf.layers.dense(inputs=self._action, units=self.n_units, activation=activation, kernel_initializer=initializer)
                for _ in range(self.n_layers - 1):
                    hidden_action = tf.layers.dense(inputs=hidden_action, units=self.n_units, activation=activation, kernel_initializer=initializer)
            # surrogate the network
            surrogate = hidden_state + hidden_action
            self._q_value = tf.layers.dense(inputs=surrogate, units=1, activation=None, kernel_initializer=initializer)
            self._q_value = tf.identity(self._q_value)
            # train the network
            self._loss = tf.reduce_mean(tf.square(self._y - self._q_value))
            self._train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)
            self._action_gradient = tf.gradients(self._q_value, self._action)

    def q_value(self, state, action):
        state = np.reshape(state, newshape=(1, 2))
        action = np.reshape(action, newshape=(1, 1))
        return self.session.run(self._q_value, feed_dict={self._state : state, self._action : action})

    def q_values(self, state_batch, action_batch):
        # examine if the sizes of two batches match
        if state_batch.shape[0] != action_batch.shape[0]:
            print("Error : The size of the batches doesn't match")
            return -1

        return self.session.run(self._q_value, feed_dict={self._state : state_batch, self._action : action_batch})

    def update(self, state_batch, action_batch, y_batch):
        return self.session.run([self._loss, self._train], feed_dict={self._state : state_batch, self._action : action_batch, self._y : y_batch})

    def get_gradient(self, state_batch, action_batch):
        return self.session.run(self._action_gradient, feed_dict={self._state : state_batch, self._action : action_batch})[0]

    def get_variables(self):
        return tf.trainable_variables(scope=self.scope)