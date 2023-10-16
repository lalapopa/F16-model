from collections import deque

import numpy as np
import tensorflow as tf



DEFAULT_RLS_GAMMA = 0.8
DEFAULT_RLS_COVARIANCE = 1
DEFAULT_RLS_CONSTANT = True

DEFAULT_HIDDEN_LAYER_SIZE = [6, 6]
DEFAULT_LR = 1.0
DEFAULT_ACTIVATION = tf.nn.tanh
DEFAULT_SESSION_CONFIG = None
DEFAULT_INCREMENTAL = False


class RecursiveLeastSquares:
    def __init__(self, **kwargs):
        # Read kwargs
        self.state_size = kwargs["state_size"]
        self.action_size = kwargs["action_size"]
        self.predict_delta = kwargs.get("predict_delta", DEFAULT_INCREMENTAL)
        self.gamma = kwargs.get("gamma", DEFAULT_RLS_GAMMA)
        self.covariance = kwargs.get("covariance", DEFAULT_RLS_COVARIANCE)
        self.constant = kwargs.get("constant", DEFAULT_RLS_CONSTANT)
        self.nb_vars = self.state_size + self.action_size
        if self.constant:
            self.nb_coefficients = self.state_size + self.action_size + 1
            self.constant_array = np.array([[[1]]])
        else:
            self.nb_coefficients = self.state_size + self.action_size

        # Initialize
        self.reset()

    def update(self, state, action, next_state):
        "Update parameters and covariance."

        state = state.reshape([-1, 1])
        action = action.reshape([-1, 1])
        next_state = next_state.reshape([-1, 1])

        if self.skip_update:
            # Store x and u
            self.x = state
            self.u = action
            self.skip_update = False
            return

        # Incremental switch
        if self.predict_delta:
            self.X[: self.state_size] = state - self.x
            self.X[self.state_size : self.nb_vars] = action - self.u
            Y = next_state - state
            # Store x and u
            self.x = state
            self.u = action
        else:
            self.X[: self.state_size] = state
            self.X[self.state_size : self.nb_vars] = action
            Y = next_state

        # Error
        Y_hat = np.matmul(self.X.T, self.W).T
        error = (Y - Y_hat).T

        # Intermidiate computations
        covX = np.matmul(self.cov, self.X)
        Xcov = np.matmul(self.X.T, self.cov)
        gamma_XcovX = self.gamma + np.matmul(Xcov, self.X)

        # Update weights and covariance
        self.W = self.W + np.matmul(covX, error) / gamma_XcovX
        self.cov = (self.cov - np.matmul(covX, Xcov) / gamma_XcovX) / self.gamma

    def predict(self, state, action):
        state = state.reshape([-1, 1])
        action = action.reshape([-1, 1])

        if self.predict_delta:
            self.X[: self.state_size] = state - self.x
            self.X[self.state_size : self.nb_vars] = action - self.u
            X_next_pred = state + np.matmul(self.W.T, self.X)
        else:
            self.X[: self.state_size] = state
            self.X[self.state_size : self.nb_vars] = action
            X_next_pred = np.matmul(self.W.T, self.X)
        return X_next_pred

    def gradient_state(self, state, action):
        gradients = self.W[: self.state_size, :].T
        if self.predict_delta:
            gradients = gradients  # + np.identity(self.state_size)
        return gradients

    def gradient_action(self, state, action):
        gradient = self.W[self.state_size : self.nb_vars, :].T
        return gradient

    def reset(self):
        "Reset parameters and covariance. Check if last state is"

        self.X = np.ones([self.nb_coefficients, 1])
        # self.W      = np.eye(self.nb_coefficients, self.state_size)
        self.W = np.zeros([self.nb_coefficients, self.state_size])
        self.cov = np.identity(self.nb_coefficients) * self.covariance

        if self.predict_delta:
            self.x = np.zeros([self.state_size, 1])
            self.u = np.zeros([self.action_size, 1])
            self.skip_update = True
        else:
            self.skip_update = False

    def reset_covariance(self):
        self.cov = np.identity(self.nb_coefficients) * self.covariance
