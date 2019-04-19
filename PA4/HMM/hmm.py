from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # translate Observation from symbol to index
        observations = np.zeros(L, dtype=int)
        for t in range(L):
            observations[t] = self.obs_dict[Osequence[t]]

        # initialization for t = 1
        z_t = observations[0]
        for s in range(S):
            alpha[s, 0] = self.pi[s] * self.B[s, z_t]

        # dynamic programing for t = 2 to T
        for t in range(1, L):
            z_t = observations[t]
            for s in range(S):
                b_sz = self.B[s, z_t]
                alpha[s, t] = b_sz * np.dot(alpha[:, t - 1], self.A[:, s])
        ###################################################
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # translate Observation from symbol to index
        observations = np.zeros(L, dtype=int)
        for t in range(L):
            observations[t] = self.obs_dict[Osequence[t]]

        # initialization for t = T
        z_t = observations[L - 1]
        for s in range(S):
            beta[s, L - 1] = 1

        # dynamic programming for t = T-1 to 1
        for t in range(L - 2, -1, -1):
            for s in range(S):
                beta[s, t] = np.sum(self.A[s, :] * self.B[:, observations[t + 1]] * beta[:, t + 1])
        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        # use forward algorithm to compute forward message
        alpha = self.forward(Osequence)

        # use forward message to compute sequence prob
        prob = np.sum(alpha[:, -1])
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        ###################################################
        seq_prob = self.sequence_prob(Osequence)

        # get forward message and backward message
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        prob = alpha * beta / seq_prob
        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        S = len(self.pi)
        L = len(Osequence)

        # translate Observation from symbol to index
        observations = np.zeros(L, dtype=int)
        for t in range(L):
            observations[t] = self.obs_dict[Osequence[t]]

        # initialization for delta (prob np.array) and Delta (state index np.array)
        delta = np.zeros(shape=(S, L))
        Delta = np.ones(shape=(S, L), dtype=int) * -1  # -1 means not decided yet or no need for previous state (t=1)

        # initialize for t = 1
        delta[:, 0] = self.B[:, observations[0]] * self.pi

        # dynamic programming for Viterbi algorithm
        for t in range(1, L):
            delta[:, t] = self.B[:, observations[t]] * np.max((self.A[:, :].T * delta[:, t - 1]).T, axis=0)
            Delta[:, t] = np.argmax((self.A[:, :].T * delta[:, t - 1]).T, axis=0)

        # get list map state index to state symbol
        idx_to_symbol = [i[0] for i in sorted(self.state_dict.items(), key=lambda kv: kv[1])]

        # find path using Delta
        path = [None for t in range(L)]
        # prev_idx = int(np.argmax(delta[:, L - 1]))
        # path[L - 1] = idx_to_symbol[prev_idx]
        self.delta = delta
        self.Delta = Delta
        path_idx = np.ones(L, dtype=int) * -1
        path_idx[L - 1] = int(np.argmax(delta[:, L - 1]))
        path[L - 1] = idx_to_symbol[path_idx[L - 1]]
        for t in range(L - 1, 0, -1):
            path_idx[t - 1] = Delta[path_idx[t], t]
            path[t - 1] = idx_to_symbol[path_idx[t - 1]]
        ###################################################
        return path
