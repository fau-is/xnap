'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

from xnap.explanation.LSTM.LRP_linear_layer import *


class LSTM_bidi:

    def __init__(self, args, model, input_encoded):
        """
        Load trained model from file.
        """

        self.args = args

        # input embedded
        self.E = input_encoded

        # model weights
        self.model = model

        """
        Assumptions:
        - bias bxh_left and bxh_right is not stored by keras
        - bias of output layer is also set to 0
        """

        # LSTM left encoder
        self.Wxh_Left = model.layers[1].get_weights()[0].T  # shape 4d*e // kernel left lstm layer // d = neurons
        # self.bxh_Left = model["bxh_Left"]  # shape 4d
        self.Whh_Left = model.layers[1].get_weights()[1].T  # shape 4d*d // recurrent kernel left lstm layer
        self.bhh_Left = model.layers[1].get_weights()[2].T  # shape 4d // biases left lstm layer

        # LSTM right encoder
        self.Wxh_Right = model.layers[1].get_weights()[3].T  # shape 4d*e // kernel right lstm layer
        # self.bxh_Right = model["bxh_Right"]
        self.Whh_Right = model.layers[1].get_weights()[4].T  # shape 4d*d // recurrent kernel right lstm layer
        self.bhh_Right = model.layers[1].get_weights()[5].T  # shape 4d // biases right lstm layer

        # linear output layer
        # note Keras does not provide two output weight vector of the bi-lslm cell; so, we divided the vector in two equal parts
        self.Why_Left = model.layers[2].get_weights()[0].T  # shape C*d
        self.Why_Left = self.Why_Left[:, 0:100]
        self.Why_Right = model.layers[2].get_weights()[0].T  # shape C*d
        self.Why_Right = self.Why_Right[:, 100:200]


    def set_input(self, w, delete_pos=None):
        """
        Build the numerical input sequence x/x_rev from the word indices w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """
        T = len(w)  # sequence length
        d = int(self.Wxh_Left.shape[0] / 4)  # hidden layer dimensions
        e = self.args.dim  # E.shape[1]   # onehot dimensions
        x = self.E

        if delete_pos is not None:
            x[delete_pos, :] = np.zeros((len(delete_pos), e))

        self.w = w
        self.x = x
        self.x_rev = x[::-1, :].copy()

        self.h_Left = np.zeros((T + 1, d))
        self.c_Left = np.zeros((T + 1, d))
        self.h_Right = np.zeros((T + 1, d))
        self.c_Right = np.zeros((T + 1, d))

    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T = len(self.w)
        d = int(self.Wxh_Left.shape[0] / 4)
        # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):     
        idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,g,f,o separately

        # initialize
        self.gates_xh_Left = np.zeros((T, 4 * d))
        self.gates_hh_Left = np.zeros((T, 4 * d))
        self.gates_pre_Left = np.zeros((T, 4 * d))  # gates pre-activation
        self.gates_Left = np.zeros((T, 4 * d))  # gates activation

        self.gates_xh_Right = np.zeros((T, 4 * d))
        self.gates_hh_Right = np.zeros((T, 4 * d))
        self.gates_pre_Right = np.zeros((T, 4 * d))
        self.gates_Right = np.zeros((T, 4 * d))

        for t in range(T):
            self.gates_xh_Left[t] = np.dot(self.Wxh_Left, self.x[t])
            self.gates_hh_Left[t] = np.dot(self.Whh_Left, self.h_Left[t - 1])
            self.gates_pre_Left[t] = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bhh_Left  # + self.bxh_Left
            self.gates_Left[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Left[t, idx]))
            self.gates_Left[t, idx_g] = np.tanh(self.gates_pre_Left[t, idx_g])
            self.c_Left[t] = self.gates_Left[t, idx_f] * self.c_Left[t - 1] + self.gates_Left[t, idx_i] * \
                             self.gates_Left[t, idx_g]
            self.h_Left[t] = self.gates_Left[t, idx_o] * np.tanh(self.c_Left[t])

            self.gates_xh_Right[t] = np.dot(self.Wxh_Right, self.x_rev[t])
            self.gates_hh_Right[t] = np.dot(self.Whh_Right, self.h_Right[t - 1])
            self.gates_pre_Right[t] = self.gates_xh_Right[t] + self.gates_hh_Right[t] + self.bhh_Right  # + self.bxh_Right
            self.gates_Right[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Right[t, idx]))
            self.gates_Right[t, idx_g] = np.tanh(self.gates_pre_Right[t, idx_g])
            self.c_Right[t] = self.gates_Right[t, idx_f] * self.c_Right[t - 1] + self.gates_Right[t, idx_i] * \
                              self.gates_Right[t, idx_g]
            self.h_Right[t] = self.gates_Right[t, idx_o] * np.tanh(self.c_Right[t])

        self.y_Left = np.dot(self.Why_Left, self.h_Left[T - 1])
        self.y_Right = np.dot(self.Why_Right, self.h_Right[T - 1])
        self.s = self.y_Left + self.y_Right

        return self.s.copy()  # prediction scores


    def lrp(self, w, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(w)
        self.forward()

        T = len(self.w)
        d = int(self.Wxh_Left.shape[0] / 4)
        e = self.args.dim  # E.shape[1]
        C = self.Why_Left.shape[0]  # number of classes
        idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,g,f,o separately

        # initialize
        Rx = np.zeros(self.x.shape)
        Rx_rev = np.zeros(self.x.shape)

        Rh_Left = np.zeros((T + 1, d))
        Rc_Left = np.zeros((T + 1, d))
        Rg_Left = np.zeros((T, d))  # gate g only
        Rh_Right = np.zeros((T + 1, d))
        Rc_Right = np.zeros((T + 1, d))
        Rg_Right = np.zeros((T, d))  # gate g only

        Rout_mask = np.zeros((C))
        Rout_mask[LRP_class] = 1.0

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh_Left[T - 1] = lrp_linear(self.h_Left[T - 1], self.Why_Left.T, np.zeros((C)), self.s, self.s * Rout_mask,
                                    2 * d, eps, bias_factor, debug=False)
        Rh_Right[T - 1] = lrp_linear(self.h_Right[T - 1], self.Why_Right.T, np.zeros((C)), self.s, self.s * Rout_mask,
                                     2 * d, eps, bias_factor, debug=False)

        for t in reversed(range(T)):
            Rc_Left[t] += Rh_Left[t]
            Rc_Left[t - 1] = lrp_linear(self.gates_Left[t, idx_f] * self.c_Left[t - 1], np.identity(d), np.zeros((d)),
                                        self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rg_Left[t] = lrp_linear(self.gates_Left[t, idx_i] * self.gates_Left[t, idx_g], np.identity(d),
                                    np.zeros((d)), self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rx[t] = lrp_linear(self.x[t], self.Wxh_Left[idx_g].T, self.bhh_Left[idx_g],  # self.bxh_Left[idx_g] +
                               self.gates_pre_Left[t, idx_g], Rg_Left[t], d + e, eps, bias_factor, debug=False)
            Rh_Left[t - 1] = lrp_linear(self.h_Left[t - 1], self.Whh_Left[idx_g].T,
                                        self.bhh_Left[idx_g], self.gates_pre_Left[t, idx_g],  # self.bxh_Left[idx_g] +
                                        Rg_Left[t], d + e, eps, bias_factor, debug=False)

            Rc_Right[t] += Rh_Right[t]
            Rc_Right[t - 1] = lrp_linear(self.gates_Right[t, idx_f] * self.c_Right[t - 1], np.identity(d),
                                         np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor,
                                         debug=False)
            Rg_Right[t] = lrp_linear(self.gates_Right[t, idx_i] * self.gates_Right[t, idx_g], np.identity(d),
                                     np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor, debug=False)
            Rx_rev[t] = lrp_linear(self.x_rev[t], self.Wxh_Right[idx_g].T,
                                   self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],  # self.bxh_Right[idx_g] +
                                   Rg_Right[t], d + e, eps, bias_factor, debug=False)
            Rh_Right[t - 1] = lrp_linear(self.h_Right[t - 1], self.Whh_Right[idx_g].T,
                                         self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],  # self.bxh_Right[idx_g] +
                                         Rg_Right[t], d + e, eps, bias_factor, debug=False)

        return Rx, Rx_rev[::-1, :], Rh_Left[-1].sum() + Rc_Left[-1].sum() + Rh_Right[-1].sum() + Rc_Right[-1].sum()
