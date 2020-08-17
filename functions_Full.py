import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from tqdm.notebook import tqdm

def chemostat_full(t, var, par, u):
    X, A, B, S, E_A, E_B, R_A, R_B = var
    D, S_0, Y_X, Y_A, Y_B, mu_0, K_S, K_A, K_B, K_EB, R_A1, R_A2, K_SA, K_AA, R_B1, R_B2, K_SB,\
        K_BB, tau_1, tau_2, tau_3, tau_4, N_A, N_B, K_HA, K_HB = par

    # Parameters
    mu = mu_0 * S * np.exp(- A/K_A - B/K_B - E_B/K_EB) / (K_S + S)
    mu_A = E_A * S * np.exp(- A/K_AA) / (K_SA + S)
    mu_B = E_B * S * np.exp(- B/K_BB) / (K_SB + S)
    R_A0 = R_A1 + (R_A2 - R_A1) * (u ** N_A) / (K_HA + u ** N_A)
    R_B0 = R_B1 + (R_B2 - R_B1) * ((1 - u) ** N_B) / (K_HB + (1 - u) ** N_B)

    out = np.zeros(8)

    # dX/dt
    out[0] = (mu - D) * X

    # dA/dt
    out[1] = mu_A * X - D * A

    # dB/dt
    out[2] = mu_B * X - D * B

    # dS/dt
    out[3] = - (mu/Y_X + mu_A/Y_A + mu_B/Y_B) * X + (S_0 - S) * D

    # dE_A/dt
    out[4] = - (1 / tau_1) * (E_A - R_A)

    # dE_B/dt
    out[5] = - (1 / tau_2) * (E_B - R_B)

    # dR_A/dt
    out[6] = - (1 / tau_3) * (R_A - R_A0)

    # dR_B/dt
    out[7] = - (1 / tau_4) * (R_B - R_B0)

    return out

def light_fun(t):
    try:
        if t<96:
            if t<48:
                return 1
            else:
                return 0
        return np.random.uniform(0, 1)
    except:
        print('time failed at:', t)
        return .5


def ODE_int(steps, t_step, t_points, par):
    pbar = tqdm(total=steps, desc='Time Stepping')
    t, X, A, B, S, E_A, E_B, R_A, R_B, u = np.zeros((10, steps, t_points))
    pbar.update()
    t_start = 0
    init = np.array([1.635e-1,
               5.159,
               2.170,
               2.668e-1,
               1.666,
               9.883e-2,
               1.7881,
               8.85e-2])

    initialize = True
    my_u = [1]
    np.random.seed(1234)
    for j in range(np.int(steps)):
        t_end = t_start + t_step


        if initialize:
            t_end = 1000
            my_u = 0.5
        pbar.update()
        t_span = np.linspace(t_start, t_end, t_points + 1, endpoint=True)
        t_max_step = t_span[1] - t_span[0]

        my_u = np.zeros(t_span.shape) + light_fun(t_start)
        if initialize:
            my_u = np.zeros(t_span.shape) + 0.5
        u[j,:] = my_u[:-1]

        sol = solve_ivp(chemostat_full, [t_start, t_end], init, t_eval=t_span, args=[par, my_u[0]], max_step = t_max_step / 10, method='BDF')
        t[j,:] = sol.t[:-1]
        X[j,:], A[j,:], B[j,:], S[j,:], E_A[j,:], E_B[j,:], R_A[j,:], R_B[j,:] = sol.y[:,:-1]

        if initialize:
            initialize = False
        else:
            t_start = t_end

        init = sol.y[:,-1]

    return t, X, A, B, S, E_A, E_B, R_A, R_B, u

class RK4_Integrator_Model(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation):

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
        self.ANNout = tf.keras.layers.Dense(4)

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

class RK4_Integrator_Model_7(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation):

        super(RK4_Integrator_Model_7, self).__init__()
        # self.k23_Layer = RK_Layer(2, h)
        # self.k4_Layer = RK_Layer(4, h)
        # self.sumLayer = sum_Layer(h)


        # self.h = h
        # Hidden Layers
        self.ANN7 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN6 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN5 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN4 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs[:, :-1]
        h = inputs[:, -1]
        h = tf.reshape(h, (-1, 1))
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs1))))))))

        t1 = inputs[:, :-2] + k1 * h / 2
        t2 = inputs[:, -2]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs2 = tf.concat([t1, t2], axis=1)

        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs2))))))))


        t1 = inputs[:, :-2] + k2 * h / 2
        t2 = inputs[:, -2]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs3 = tf.concat([t1, t2], axis=1)

        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs3))))))))

        t1 = inputs[:, :-2] + k3 * h
        t2 = inputs[:, -2]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs4 = tf.concat([t1, t2], axis=1)


        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs4))))))))

        # RK4 Output for Prediction

        outputs = inputs[:,:-2] + (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)


        return outputs

class RK4_Integrator_Model_TE(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation, time_embed):

        super(RK4_Integrator_Model_TE, self).__init__()
        # self.k23_Layer = RK_Layer(2, h)
        # self.k4_Layer = RK_Layer(4, h)
        # self.sumLayer = sum_Layer(h)


        self.time_embed = time_embed
        # Hidden Layers
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs[:, :-1]
        h = inputs[:, -1]
        h = tf.reshape(h, (-1, 1))
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs1))))
        t0 = (selected_inputs1[:,:-5] + selected_inputs1[:,5:]) / 2
        t1 = selected_inputs1[:, -5:-1] + k1 * h / 2
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs2 = tf.concat([t0, t1, t2], axis=1)
        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs2))))

        t0 = (selected_inputs1[:,:-5] + selected_inputs1[:,5:]) / 2
        t1 = selected_inputs1[:, -5:-1] + k2 * h / 2
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs3 = tf.concat([t0, t1, t2], axis=1)

        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs3))))

        t0 = selected_inputs1[:,5:]
        t1 = selected_inputs1[:, -5:-1] + k3 * h
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs4 = tf.concat([t0, t1, t2], axis=1)


        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(selected_inputs4))))

        # RK4 Output for Prediction

        outputs = selected_inputs1[:,-5:-1] + (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)


        return outputs

class RK4_Integrator_Model_TE_7(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation, time_embed):

        super(RK4_Integrator_Model_TE_7, self).__init__()
        # self.k23_Layer = RK_Layer(2, h)
        # self.k4_Layer = RK_Layer(4, h)
        # self.sumLayer = sum_Layer(h)


        self.time_embed = time_embed
        # Hidden Layers
        self.ANN7 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN6 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN5 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN4 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs[:, :-1]
        h = inputs[:, -1]
        h = tf.reshape(h, (-1, 1))
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs1))))))))
        t0 = (selected_inputs1[:,:-5] + selected_inputs1[:,5:]) / 2
        t1 = selected_inputs1[:, -5:-1] + k1 * h / 2
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs2 = tf.concat([t0, t1, t2], axis=1)
        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs2))))))))

        t0 = (selected_inputs1[:,:-5] + selected_inputs1[:,5:]) / 2
        t1 = selected_inputs1[:, -5:-1] + k2 * h / 2
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs3 = tf.concat([t0, t1, t2], axis=1)

        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs3))))))))

        t0 = selected_inputs1[:,5:]
        t1 = selected_inputs1[:, -5:-1] + k3 * h
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs4 = tf.concat([t0, t1, t2], axis=1)


        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs4))))))))

        # RK4 Output for Prediction

        outputs = selected_inputs1[:,-5:-1] + (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)


        return outputs


class RK4_Integrator_Model_TE_7_CL(tf.keras.Model):
    def __init__(self, HL_Nodes, Activation, time_embed):

        super(RK4_Integrator_Model_TE_7, self).__init__()
        # self.k23_Layer = RK_Layer(2, h)
        # self.k4_Layer = RK_Layer(4, h)
        # self.sumLayer = sum_Layer(h)


        self.time_embed = time_embed
        # Hidden Layers
        self.ANN7 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN6 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN5 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN4 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN3 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN2 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)
        self.ANN1 = tf.keras.layers.Dense(HL_Nodes, activation=Activation)

        #Output of Dense Layers
        self.ANNout = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        selected_inputs1 = inputs[:, :-1]
        h = inputs[:, -1]
        h = tf.reshape(h, (-1, 1))
        # First Pass
        k1 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs1))))))))
        t0 = (selected_inputs1[:,:-5] + selected_inputs1[:,5:]) / 2
        t1 = selected_inputs1[:, -5:-1] + k1 * h / 2
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs2 = tf.concat([t0, t1, t2], axis=1)
        # Second Pass
        k2 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs2))))))))

        t0 = (selected_inputs1[:,:-5] + selected_inputs1[:,5:]) / 2
        t1 = selected_inputs1[:, -5:-1] + k2 * h / 2
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs3 = tf.concat([t0, t1, t2], axis=1)

        # Third Pass
        k3 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs3))))))))

        t0 = selected_inputs1[:,5:]
        t1 = selected_inputs1[:, -5:-1] + k3 * h
        t2 = selected_inputs1[:, -1]
        t2 = tf.reshape(t2, (-1, 1))
        selected_inputs4 = tf.concat([t0, t1, t2], axis=1)


        # Fourth Pass
        k4 = self.ANNout(
                 self.ANN1(
                 self.ANN2(
                 self.ANN3(
                 self.ANN4(
                 self.ANN5(
                 self.ANN6(
                 self.ANN7(
                     selected_inputs4))))))))

        # RK4 Output for Prediction

        outputs = selected_inputs1[:,-5:-1] + (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)


        return outputs
