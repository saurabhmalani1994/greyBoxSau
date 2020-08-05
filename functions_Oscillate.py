import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from tqdm.notebook import tqdm

def chemostat_simple(t, var, par, mult):
    m1, m2, m3, P1, P2, P3 = var
    m = [m1, m2, m3]
    P = [P1, P2, P3]
    alpha, alpha_0, Beta, n = par

    out = np.zeros(6)
    Beta = Beta * mult
    # dmi/dt
    for i in range(3):
        out[i] = alpha / ((1 + P[(i-1) % 3] ** n)) + alpha_0 - m[i]
        out[i + 3] = Beta[i] * (m[i] - P[i])

    return out

def ODE_int(datasets, steps, t_step, t_points, par):
    pbar = tqdm(total=datasets+1, desc='Datasets')
    pbar2 = tqdm(total=steps, desc='Time Stepping')
    t = np.zeros((datasets + 1, steps, t_points))
    m1 = np.zeros((datasets + 1, steps, t_points))
    m2 = np.zeros((datasets + 1, steps, t_points))
    m3 = np.zeros((datasets + 1, steps, t_points))
    P1 = np.zeros((datasets + 1, steps, t_points))
    P2 = np.zeros((datasets + 1, steps, t_points))
    P3 = np.zeros((datasets + 1, steps, t_points))
    t_mult = np.zeros((datasets + 1, steps, t_points))
    mult_plot = np.zeros((datasets + 1, steps, t_points))
    for i in range(datasets + 1):
        pbar.update()
        t_start = 0
        # t_end = t_step
        # init = [1, 1, 1]
        # init = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        init = [np.random.uniform(0,50), np.random.uniform(0,50), np.random.uniform(0,50),
                np.random.uniform(0,50), np.random.uniform(0,50), np.random.uniform(0,50)]
        pbar2.reset()
        my_Mult = np.random.choice([10,100,1000])
        for j in range(np.int(steps)):
            # mult_mult = 2 ** np.random.uniform(-1,1)
            # my_Mult = np.max([np.min([my_Mult * mult_mult, 1000]), 10])
            my_Mult = 10
            t_end = t_start + t_step / 2 ** np.log10(1/my_Mult)
            pbar2.update()
            t_span = np.linspace(t_start, t_end, t_points, endpoint=False)
            t_max_step = t_span[1] - t_span[0]
            mult = np.zeros(t_span.shape) + my_Mult
            # mult = [1]
            t_mult[i,j,:] = t_span
            mult_plot[i,j,:] = mult

            sol = solve_ivp(chemostat_simple, [t_start, t_end], init, t_eval=t_span, args=[par, mult[0]], max_step = t_max_step / 5, method='BDF')
            t1 = sol.t
            m11, m21, m31, P11, P21, P31 = sol.y

            t[i,j,:] = t1
            m1[i,j,:] = m11
            m2[i,j,:] = m21
            m3[i, j, :] = m31
            P1[i, j, :] = P11
            P2[i, j, :] = P21
            P3[i, j, :] = P31


            t_start = t_end
            # t_end += t_step
            init = [m11[-1], m21[-1], m31[-1], P11[-1], P21[-1], P31[-1]]

    return m1, m2, m3, P1, P2, P3, mult_plot, t, t_mult

class RK4_Integrator_Model(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation, learning_rate):

        super(RK4_Integrator_Model, self).__init__()
        # self.k23_Layer = RK_Layer(2, h)
        # self.k4_Layer = RK_Layer(4, h)
        # self.sumLayer = sum_Layer(h)


        # self.h = h
        # Hidden Layers
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(6)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs[:, :-1]
        h = inputs[:, -1]
        h = tf.reshape(h, (-1, 1))
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs1))))

        t1 = inputs[:, :-2] + k1 * h / 2
        t2 = inputs[:, -2]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs2 = tf.concat([t1, t2], axis=1)

        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs2))))


        t1 = inputs[:, :-2] + k2 * h / 2
        t2 = inputs[:, -2]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs3 = tf.concat([t1, t2], axis=1)

        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs3))))

        t1 = inputs[:, :-2] + k3 * h
        t2 = inputs[:, -2]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs4 = tf.concat([t1, t2], axis=1)


        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs4))))

        # RK4 Output for Prediction

        outputs = inputs[:,:-2] + (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)


        return outputs