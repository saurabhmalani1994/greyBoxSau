import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf

def chemostat_simple(t, var, par, E0):
    S, E = var
    S0, theta, k_cat, K_m = par

    out = np.zeros(2)

    # dS/dt
    out[0] = (1 / theta) * (S0 - S) - (k_cat * E * S) / (K_m + S)

    # dE/dt
    out[1] = (1 / theta) * (E0 - E)

    return out

def ODE_int(datasets, t_max, t_step, par):

    # t, S, E, t_E0, E0_plot = [[] for i in range(datasets + 1)], [[] for i in range(datasets + 1)],\
    #                          [[] for i in range(datasets + 1)], [[] for i in range(datasets + 1)],\
    #                          [[] for i in range(datasets + 1)]
    t, S, E, t_E0, E0_plot = [], [], [], [], []
    for i in range(datasets + 1):
        t_start = 0
        t_end = t_step
        S_init = np.random.uniform(0.02, 0.025)
        E_init = np.random.uniform(0, 0.01)
        init = [S_init, E_init]
        while t_end <= t_max:
            t_span = np.linspace(t_start, t_end, 100, endpoint=False)
            E0 = np.random.uniform(0, 0.01)

            t_E0[i] = np.append(t_E0[i], t_start)
            t_E0[i] = np.append(t_E0[i], t_end)
            E0_plot[i] = np.append(E0_plot[i], E0)
            E0_plot[i] = np.append(E0_plot[i], E0)

            t_start = t_span[0]
            t_end = t_span[-1]

            sol = solve_ivp(chemostat_simple, [t_start, t_end], init, t_eval=t_span, args=[par, E0], method='BDF')
            t1 = sol.t
            S1, E1 = sol.y

            t[i] = np.append(t[i], t1)
            S[i] = np.append(S[i], S1)
            E[i] = np.append(E[i], E1)

            t_start += t_step
            t_end += t_step
            init = [S1[-1], E1[-1]]

    return S, E, E0_plot, t, t_E0

def ODE_int2(datasets, t_max, t_step, t_points, par):
    # t_points = 100
    y =  t_points * np.int((t_max / t_step))
    t = np.zeros((datasets + 1, np.int((t_max / t_step)), t_points))
    S = np.zeros((datasets + 1, np.int((t_max / t_step)), t_points))
    E = np.zeros((datasets + 1, np.int((t_max / t_step)), t_points))
    t_E0 = np.zeros((datasets + 1, np.int((t_max / t_step)), t_points))
    E0_plot = np.zeros((datasets + 1, np.int((t_max / t_step)), t_points))

    for i in range(datasets + 1):
        t_start = 0
        t_end = t_step
        S_init = np.random.uniform(0.02, 0.025)
        E_init = np.random.uniform(0, 0.01)
        init = [S_init, E_init]

        for j in range(np.int((t_max / t_step))):
            t_span = np.linspace(t_start, t_end, t_points, endpoint=False)
            E0 = np.zeros(t_span.shape) + np.random.uniform(0, 0.01)

            t_E0[i,j,:] = t_span
            E0_plot[i,j,:] = E0

            # t_E0[i,j,:] = [t_start, t_end]
            # E0_plot[i,j,:] = [E0[0], E0[-1]]

            sol = solve_ivp(chemostat_simple, [t_start, t_end], init, t_eval=t_span, args=[par, E0[0]], method='BDF')
            t1 = sol.t
            S1, E1 = sol.y

            t[i,j,:] = t1
            S[i,j,:] = S1
            E[i,j,:] = E1

            t_start += t_step
            t_end += t_step
            init = [S1[-1], E1[-1]]

    return S, E, E0_plot, t, t_E0

class RK4_Integrator():
    def __init__(self, HL_Nodes, Activation, learning_rate, h):

        #Input
        inputs = tf.keras.Input(shape=(2,))

        # Hidden Layers
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(1)

        selected_inputs1 = inputs
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs1))))

        # k1_add = tf.concat([k1,[[0]]],1)
        # selected_inputs2 = tf.math.add(inputs,k1_add * h / 2)
        # # Second Pass
        # k2 = self.ANNout(
        #          self.ANN1(
        #          self.ANN2(
        #          self.ANN3(selected_inputs2))))
        #
        # k2_add = tf.concat([k2, [[0]]], 1)
        # selected_inputs3 = tf.math.add(inputs, k2_add * h / 2)
        #
        # # Third Pass
        # k3 = self.ANNout(
        #          self.ANN1(
        #          self.ANN2(
        #          self.ANN3(selected_inputs3))))
        #
        # k3_add = tf.concat([k3, [[0]]], 1)
        # selected_inputs4 = tf.math.add(inputs, k3_add * h / 2)
        #
        # # Fourth Pass
        # k4 = self.ANNout(
        #          self.ANN1(
        #          self.ANN2(
        #          self.ANN3(selected_inputs4))))
        #
        # k4_add = tf.concat([k4, [[0]]], 1)
        #
        # # RK4 Output for Prediction
        # outputs = inputs + (1/6) * h * (k1_add + 2 * k2_add + 2 * k3_add + k4_add)

        outputs = k1

        # define model--inherits all Keras model methods (see keras docs for details
        self.BB_model= tf.keras.Model(inputs=[inputs],outputs=outputs)

        self.BB_model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                                 loss = tf.compat.v1.losses.mean_squared_error)

    def fit(self,x=None, y=None, batch_size=None, epochs=1,verbose=1,callbacks=None,
            validation_data=None, validation_split=None, shuffle=True, class_weight=None, sample_weight=None,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None):

        return self.BB_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                 verbose=verbose,callbacks=callbacks, validation_data=validation_data, validation_split=validation_split,
                 shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                 initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                 validation_steps=validation_steps)

    def predict(self, input):
        output = self.BB_model([input])
        return output

class RK4_Integrator_Model(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation, learning_rate, h):
        super(RK4_Integrator_Model, self).__init__()
        self.h = h
        # Hidden Layers
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(3)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs1))))

        selected_inputs2 = inputs + k1 * self.h / 2
        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs2))))


        selected_inputs3 = inputs + k2 * self.h / 2
        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs3))))

        selected_inputs4 = inputs + k3 * self.h

        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs4))))

        # RK4 Output for Prediction
        outputs = inputs + (1/6) * self.h * (k1 + 2 * k2 + 2 * k3 + k4)

        # outputs = k1

        return outputs

class RK4_Integrator_Model2(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation, learning_rate, h):

        super(RK4_Integrator_Model2, self).__init__()
        # self.k23_Layer = RK_Layer(2, h)
        # self.k4_Layer = RK_Layer(4, h)
        # self.sumLayer = sum_Layer(h)


        self.h = h
        # Hidden Layers
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs1))))

        t1 = tf.reshape(inputs[:, 0],[-1,1]) + k1 * self.h / 2
        t2 = inputs[:, 1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs2 = tf.concat([t1, t2], axis=1)

        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs2))))


        t1 = tf.reshape(inputs[:, 0],[-1,1]) + k2 * self.h / 2
        t2 = inputs[:, 1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs3 = tf.concat([t1, t2], axis=1)

        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs3))))

        t1 = tf.reshape(inputs[:, 0],[-1,1]) + k3 * self.h
        t2 = inputs[:, 1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs4 = tf.concat([t1, t2], axis=1)


        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs4))))

        # RK4 Output for Prediction

        outputs = tf.reshape(inputs[:, 0],[-1,1]) + (1/6) * self.h * (k1 + 2 * k2 + 2 * k3 + k4)


        return outputs
