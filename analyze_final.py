import copy
import matplotlib.pyplot as plt
from mytools import *
import numpy as np
import boundingbox
from estimateOmega import estimate_LLS, estimate_kabsch, estimate_rotation_B
from associationdata import rotation_association
import pickle
import scipy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def get_dimensions(p1, p, q):
    p1_to_p = p - p1
    R = quat2rotm(q)
    p1_debris_frame = R.T @ p1_to_p
    L = 2*p1_debris_frame[0]
    W = 2*p1_debris_frame[1]
    D = 2*p1_debris_frame[2]
    return L, W, D

def sum_of_sinusoids(t_fit, *params_fit):
    y_fit = np.zeros_like(t_fit)
    num_sinusoids = (len(params_fit) - 0) // 4
    for i in range(num_sinusoids):
        A = params_fit[4 * i]
        omega = params_fit[4 * i + 2]
        phi = params_fit[4 * i + 1]
        C = params_fit[4 * i + 3]
        y_fit += A * np.sin(omega * t_fit + phi) + C
    return y_fit


def remove_bias(start_t, dt, y, estimated, num_sinusoids, freq_threshold, freq_skip, true, params_ini):

    nframes = len(y)
    time_interval = (nframes - 1) * dt
    y_orig = y.copy()
    t = np.linspace(start=start_t, stop=start_t + time_interval, num=nframes)
    y = y - estimated

    # Compute the FFT
    y_fft = np.fft.fft(y)
    freq = np.fft.fftfreq(nframes, d=t[1] - t[0])
    y_fft = y_fft[freq > freq_threshold]
    freq = freq[freq > freq_threshold]

    # Compute the magnitudes of the FFT
    magnitudes = np.abs(y_fft)

    # Only consider the positive frequencies (first half of the FFT result)
    positive_frequencies = freq
    positive_magnitudes = magnitudes

    # Find the peaks in the FFT magnitude spectrum
    peaks, _ = find_peaks(positive_magnitudes)

    # Extract peak magnitudes and their corresponding frequencies
    peak_magnitudes = positive_magnitudes[peaks]
    peak_frequencies = positive_frequencies[peaks]

    indices = np.arange(-1, -num_sinusoids * freq_skip - freq_skip, -freq_skip)
    top_peak_indices = np.argsort(peak_magnitudes)[indices[::-1]][::-1]

    # Extract the top three peak frequencies and their magnitudes
    top_frequencies = peak_frequencies[top_peak_indices]
    top_magnitudes = peak_magnitudes[top_peak_indices]

    initial_amplitude = max(y) - min(y)
    initial_phase = 0
    initial_constant = np.mean(y)
    initial_frequencies = 2 * np.pi * top_frequencies

    initial_guess = []
    if len(params_ini) == 0:
        for index in range(0, num_sinusoids):
            initial_guess.append(initial_amplitude)
            initial_guess.append(initial_phase)
            initial_guess.append(initial_frequencies[index])
            initial_guess.append(initial_constant)
    else:
        initial_guess = params_ini

    # Perform the curve fitting
    params, params_covariance = curve_fit(sum_of_sinusoids, t, y, p0=initial_guess)
    constant = max(sum_of_sinusoids(t, *params))
    # print(constant)

    if True:
    # if False:
        # Plot the frequency spectrum
        plt.figure()
        plt.plot(positive_frequencies, positive_magnitudes, label='Frequency spectrum')
        plt.plot(top_frequencies, top_magnitudes, 'ro', label='Top peaks')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()

        # with respect to true
        plt.figure()
        plt.scatter(t, y_orig - true, s=3, label='Data')
        # plt.plot(t, sum_of_sinusoids(t, *params))
        plt.plot(t, sum_of_sinusoids(t, *params) + estimated - true, label='Fitted Sum of Sinusoids', color='red')
        # plt.scatter(t, y_orig - sum_of_sinusoids(t, *params) - true + constant, s=3, label='Bias corrected', color='orange')
        plt.plot(t, estimated - true, label='Estimated', linestyle='-', color='Orange')
        plt.plot(t, np.zeros_like(true), label='True', linestyle='--', color='Green')
        plt.xlabel('Times (s)')
        plt.ylabel('Position with Respect to Ground Truth (m)')
        plt.legend()

        # with respect to estimated
        plt.figure()
        plt.scatter(t, y_orig - estimated, s=3, label='Data')
        plt.plot(t, sum_of_sinusoids(t, *params), label='Fitted Sum of Sinusoids', color='red')
        # plt.scatter(t, y_orig - sum_of_sinusoids(t, *params) - estimated + constant, s=3, label='Bias corrected', color='orange')
        plt.plot(t, true - estimated, label='True', linestyle='--', color='green')
        plt.plot(t, np.zeros_like(estimated), label='Estimated', linestyle='-', color='orange')
        plt.xlabel('Times (s)')
        plt.ylabel('Position with Respect to Estimated (m)')

        plt.legend()
        #
        # print(np.mean(y_orig - true))
        # print(np.mean(y_orig - (sum_of_sinusoids(t, *params)) - true))
        # print(np.mean(y_orig - sum_of_sinusoids(t, *params) - estimated))


        #
        # plt.figure()
        # t = np.arange(0., 1000., .05)
        # plt.plot(t, sum_of_sinusoids(t, *params))
        # plt.show()

    return params, constant


def correct_bias(z_p_k_meas, curr_i, dt_here, parameters, constants, R_i_L_to_B, R_i_B_to_L):
    # correct bias
    bias_z = sum_of_sinusoids(curr_i * dt_here, *parameters[2])
    bias_y = sum_of_sinusoids(curr_i * dt_here, *parameters[1])
    bias_x = sum_of_sinusoids(curr_i * dt_here, *parameters[0])
    z_p_k_B = R_i_L_to_B @ z_p_k_meas
    z_p_k_B[2] = z_p_k_B[2] + constants[2]
    # z_p_k_B[0] = z_p_k_B[0] - bias_x
    # z_p_k_B[1] = z_p_k_B[1] - bias_y
    z_p_k_L = R_i_B_to_L @ z_p_k_B

    return z_p_k_L


def drawrectangle(ax, p1, p2, p3, p4, p5, p6, p7, p8, color, linewidth, label):
    # z1 plane boundary
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=color, linewidth=linewidth)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color, linewidth=linewidth)
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color=color, linewidth=linewidth)

    # z1 plane boundary
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=color, linewidth=linewidth)
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=color, linewidth=linewidth)
    ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color=color, linewidth=linewidth)

    # Connecting
    ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=color, linewidth=linewidth)
    ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=color, linewidth=linewidth)
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=color, linewidth=linewidth, label=label)

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='#9b42f5')


def skew(vector):
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def rodrigues(omega, dt):
    e_omega = omega / np.linalg.norm(omega)  # Unit vector along omega
    phi = np.linalg.norm(omega) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R


def verticeupdate(dt, x_k):
    # Decompose the state vector
    p_k = x_k[:3]
    v_k = x_k[3:6]
    omega_k = x_k[6:9]
    p1_k = x_k[9:12]

    # Rotation matrix - rodrigues formula
    R_k_kp1 = rodrigues(omega_k, dt)
    R_k_kp1 = R_k_kp1

    # Translate vertex to origin
    p1_ko = p1_k - p_k

    # Rotate vertices
    p1_kp1o = np.matmul(R_k_kp1, p1_ko.reshape(len(p1_ko), 1))

    # Translate vertex back to new expected origin
    p1_kp1 = (p1_kp1o.T + p_k + v_k * dt).ravel()

    return p1_kp1, R_k_kp1


def orientationupdate(dt, x_k):
    # Decompose the state vector
    omega_k = x_k[6:9]
    q_k = x_k[12:16]
    qw = q_k[0];
    qx = q_k[1];
    qy = q_k[2];
    qz = q_k[3]

    # hamilton = np.array([-omega_k[0] * q_k[1] - omega_k[1] * q_k[2] - omega_k[2] * q_k[3],
    #             omega_k[0] * q_k[0] + omega_k[2] * q_k[2] - omega_k[1] * q_k[3],
    #             omega_k[1] * q_k[0] - omega_k[2] * q_k[1] + omega_k[0] * q_k[3],
    #             omega_k[2] * q_k[0] + omega_k[1] * q_k[1] - omega_k[0] * q_k[2]])

    dqkdt = 0.5 * np.array([[-qx, -qy, -qz],
                            [qw, qz, -qy],
                            [-qz, qw, qx],
                            [qy, -qx, qw]]) @ omega_k

    q_kp1 = normalize_quat(dqkdt * dt + q_k)
    q_kp1_pos = q_kp1  # if q_kp1[0] >=0 else -q_kp1
    return q_kp1_pos


def get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, q_ini):

    Rot_0 = quat2rotm(q_ini)
    Rot_0 = np.eye(3)
    # print(Rot_0)
    q_s = []
    for i in range(len(debris_pos)):

        # get rotation matrix for that timestep
        Rot_i = rodrigues(omega_true, dt * i)
        q_i = rotm2quat(Rot_i @ Rot_0)
        if i == 0:
            q_s.append(q_i)
        else:
            q_i_alt = -q_i
            q_prev = q_s[i-1]
            if np.linalg.norm(q_i - q_prev) < np.linalg.norm(q_i_alt - q_prev):
                q_s.append(q_i)
            else:
                q_s.append(q_i_alt)
    return q_s


# initialize debris position, velocity and orientation
O_B = np.array([0, 0, 0])
O_L = np.array([0, 0, 0])

with open('sim_kompsat_journal.pickle', 'rb') as sim_data:
    # with open('sim_kompsat_neg_om_longer.pickle', 'rb') as sim_data:
    # with open('sim_new_conditions.pickle', 'rb') as sim_data:
    data = pickle.load(sim_data)
XBs = data[0]
YBs = data[1]
ZBs = data[2]
PBs = data[3]
VBs = data[4]

debris_pos = data[5]
debris_vel = data[6]
Rot_L_to_B = data[7]
Rot_B_to_L = [np.transpose(r) for r in Rot_L_to_B]
omega_L = data[8]
dt = data[9]

# Estimation Loop
XLs = []  # store point cloud x in L
YLs = []
ZLs = []
PLs = []  # store x, y, z point cloud in L
VLs = VBs  # store velocity point cloud
x_s = []  # store states over time
z_s = []  # store measurements over time
P_s = []  # store covariances in time
errors = [0]
nframes = len(VBs)

# Running the simulation - Initializations

# Initializations in L Frame
vT_0 = [0.1, 0.1, 0.1]  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
omega_0 = [-1, 0, 1.]
omega_true = [-0.5, 0.3, 1.]
q_ini = [1., 0., 0., 0.]
q_true = np.array(get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, q_ini))
q_true_alt = -q_true

p_0 = np.array([-180., -320., -10.])
p1_0 = p_0 + np.array([-2 / 2, -2 / 2, -2 / 2])
x_0 = np.hstack([p_0, vT_0, omega_0, p1_0, q_ini])
num_states = len(x_0)

# Initial covariance
P_0 = np.diag([0.25, 0.5, 0.25, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25,
               0.25])  # Initial Covariance matrix

# Process noise covariance matrix
qp = 0.00000001
qv = 0.00000005
qom = 0.00000005
qp1 = 0.05
qq = 0.000005
Q = np.diag([qp, qp, qp, qv, qv, qv, qom, qom, qom, qp1, qp1, qp1, qq, qq, qq, qq])

# Measurement noise covariance matrix
p = 0.005
om = 0.5
p1 = 1
q = 0.004
R1 = np.diag([p, p, p, om, om, om, p1, p1, p1, q, q, q, q])
R2 = np.diag([p, p, p, om, om, om, p1, p1, p1])

z_q_k_1_previous = np.zeros((4,))
z_q_k_2_previous = np.zeros((4,))
# q_km1 = np.zeros((4,))

# Measurement matrix
H1 = np.zeros([len(P_0)-3, len(P_0)])  # no measuring of velocity
H1[0:3,0:3] = np.eye(3)
H1[3:,6:] = np.eye(10)
bad_attitude_measurement_flag = False
adapt = False

H2 = np.zeros([9,16])
H2[0:3,0:3] = np.eye(3)
H2[3:6,6:9] = np.eye(3)
H2[6:, 9:12] = np.eye(3)

# Kabsch estimation parameters
n_moving_average = 200
settling_time = 200
# Record keeping for angular velocity estimate
omegas_kabsch_b = np.zeros((nframes, 3))
omegas_lls_b = np.zeros((nframes, 3))
omega_kabsch_b_box = np.zeros((n_moving_average, 3))

# Get Final measurement vectors
q_kp1s = []
z_p_s = [p_0]
zv_mags = []
z_omegas = []
z_v_s = []
z_s = []
z_rans = []
z_pcas = []
x_s = [x_0]
P_s = []
original_pos_meas = []
estimated_pos = [p_0]
rotation_errors = [0]

# bias config
interval_time = 0  # for bias part
done = 0
params_x = []
params_y = []
params_z = []
centroids_inB = []
true_pos_inB = []
q_kp1s =[]
metrics = []
z_s_all = []
without_correction = []
bbox1_dimensions =[]
bbox2_dimensions = []
bbox3_dimensions = [2*(p_0 - p1_0)]

# ukf weight values
alpha = 1e-1
beta = 2
kappa = 0
# cholesky decomposition to calc sigmapoints
epsilon = 1e-10  # constant to ensure positive defineteness
dimL = len(x_0)
lambd = alpha ** 2 * (dimL + kappa) - dimL
w_0_m = lambd / (lambd + dimL)  # first weight for computing the mean
w_j_m = 0.5 / (lambd + dimL)  # consequent weights for computing the mean
w_0_c = w_0_m + (1 - alpha ** 2 + beta)  # first weight for computing covariance
w_j_c = w_j_m
tolerance = 1e-2  # threshold to which the ISPKF iterates, i.e., iterate until difference between states is below threshold

# barfoot
# lambd = 2  # formerly kappa
# w_0_m = lambd / (lambd + dimL) # first weight for computing the mean
# w_j_m = 0.5 / (lambd + dimL)  # consequent weights for computing the mean
# w_0_c = w_0_m  # first weight for computing covariance
# w_j_c = w_j_m

# counters for adaptation 12, 12, 8
pred1 = 0
pred2 = 0
pred3 = 0  # not used currently
pred4 = 0
pred5 = 0
pred6 = 0
pred7 = 0
pred8 = 0
pred9 = 0
pred10 = 0
pred11 = 0
pred12 = 0
pred13 = 0
ran1 = 0
ran2 = 0
ran3 = 0
ran4 = 0
ran5 = 0
ran6 = 0
ran7 = 0
ran8 = 0
pca1= 0
pca2 = 0
pca3 = 0
pca4 = 0
pca5 = 0
pca6 = 0
pca7 = 0
pca8 = 0
pca9 = 0
pca10 = 0
pca11 = 0
pca12 = 0
ransac = 0
prediction = 0
pca = 0
true_pred = 0
true_ran = 0
true_pca = 0

for i in range(nframes):

    print(i)
    # visualize_flag = i>40*20 and i%20==0
    visualize_flag = False

    # if i > 200:
    #     tolerance = 1e-1


    # Use first measurements for initializations of states - not implemented currently, just chose initial states up top
    if i > 0:
        # state vector as mean for sigmapoint transform
        mu_sp = x_k.copy()

        # covariance matrix for sigmapoint transform
        sigma_zz = P_k.copy()

        # cholesky decomposition for lower triangular matrix
        try:
            L = scipy.linalg.cholesky(sigma_zz, lower=True)
        except np.linalg.LinAlgError:  # happens when the diagonal is zero but numerically speaking has negative elements
            np.fill_diagonal(sigma_zz, sigma_zz.diagonal() + epsilon)
            L = scipy.linalg.cholesky(sigma_zz, lower=True)

        # initial sigmapoint
        sp_0 = mu_sp

        # other sigmapoints
        sp_s = [sp_0]
        sqrt_term = np.sqrt(dimL + lambd)
        for idx in range(0, dimL):
            col_i_L = L[:, idx]
            sp_i = mu_sp + sqrt_term * col_i_L
            sp_s.append(sp_i)
        for idx in range(0, dimL):
            col_i_L = L[:, idx]
            sp_i_L = mu_sp - sqrt_term * col_i_L
            sp_s.append(sp_i_L)

        # pass each point through prediction model
        x_kp1 = np.zeros((num_states,))
        sp_kp1s = []
        for jdx, sp in enumerate(sp_s):

            # Decompose the state vector
            p_k = sp[:3]
            v_k = sp[3:6]
            omega_k = sp[6:9]
            p1_k = sp[9:12]
            q_k = sp[12:]

            ##############
            # Prediction
            ##############

            # Position update
            p_kp1 = v_k * dt + p_k

            # Velocity update
            v_kp1 = v_k.copy()

            # Angular velocity update
            omega_kp1 = omega_k.copy()

            # Vertex update
            p1_kp1, R_k_kp1 = verticeupdate(dt, sp)

            # Orientation Update
            if i == 1:
                q_kp1s.append(q_ini)

            q_kp1 = orientationupdate(dt, sp)
            q_kp1s.append(q_kp1)

            sp_kp1_jdx = np.hstack([p_kp1, v_kp1, omega_kp1, p1_kp1, q_kp1]).ravel()
            sp_kp1s.append(sp_kp1_jdx)

            # weighted sum of sigma points to get updated state
            if jdx == 0:
                x_kp1 += w_0_m * sp_kp1_jdx
            else:
                x_kp1 += w_j_m * sp_kp1_jdx

        z_v_s.append(x_k[3:6])

        # update covariance
        P_kp1 = np.zeros((num_states, num_states))
        for kdx, sp in enumerate(sp_kp1s):
            diff = sp - x_kp1
            if kdx == 0:
                P_kp1 += w_0_c * np.outer(diff, diff.T)
            else:
                P_kp1 += w_j_c * np.outer(diff, diff.T)

        # add proces noise
        P_kp1 += Q

        # try and smooth out covariance off diagonals to ensure symmetry
        P_kp1 = 0.5 * P_kp1 + 0.5 * P_kp1.T

    #######################
    # Measurements
    #######################

    PLs.append((Rot_L_to_B[i].T @ (PBs[i]).T).T)
    # find bounding box from points
    XLs.append(PLs[i][:, 0])
    YLs.append(PLs[i][:, 1])
    ZLs.append(PLs[i][:, 2])
    X_i = XLs[i]
    Y_i = YLs[i]
    Z_i = ZLs[i]

    # Return bounding box and centroid estimate of bounding box
    z_pi_k_1, z_p_k_1, R_1 = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
    z_pi_k_2, z_p_k_2, R_1_2, normal_vecs, ranking = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, True, False)

    ############
    # bias removal
    ############

    original_pos_meas.append(z_p_k_1)
    centroids_inB.append(Rot_L_to_B[i] @ z_p_k_1)
    true_pos_inB.append(Rot_L_to_B[i] @ debris_pos[i, :])

    curr_t = i * dt
    t_start = 20  # when the first bias calculation should be initiated
    t_interval = 20  # how many seconds of data should be collected each time

    # grab data every interval
    if curr_t >= (t_start + t_interval):
        if (curr_t + t_start) % t_interval == 0 and done == 0:  # grab new data
            interval_time = curr_t - t_interval
            z_in_b = [Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(original_pos_meas)]
            z = np.array(z_in_b)
            z = z[int(interval_time / dt):, :]
            estimated_inB = np.array([Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(estimated_pos)])
            estimated = np.array(estimated_inB)
            estimated = estimated[int(interval_time / dt):, :]
            true_inB = np.array([Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(debris_pos)])
            true = np.array(true_inB)
            true = true[int(interval_time / dt):int((interval_time + t_interval) / dt) + 1, :]

            thresh = 0.05  # initial threshold to remove frequencies obtained from crosstalk with baseband frequency
            num_sin = 2  # number of sinusoids to use to fit the data
            skip = 1  # when choosing frequencies from frequency according to decreasing magnitude, skips this many frequencies
            params_z, constant_z = remove_bias(interval_time, dt, z[:, 2], estimated[:, 2], num_sin, thresh, skip, true[:, 2], params_z)
            # params_x, constant_x = remove_bias(interval_time, dt, z[:, 0], estimated[:, 0], num_sin, thresh, skip, true[:, 0], params_x)
            # params_y, constant_y = remove_bias(interval_time, dt, z[:, 1], estimated[:, 1], num_sin, thresh, skip, true[:, 1], params_y)
            parameters = [params_x, params_y, params_z]
            print(params_z)
            print(constant_z)

            constants = [0, 0, constant_z]
            done = 1

    #####################

    # Orientation association
    # R_1 is obtained from bounding box
    if i == 0:
        z_q_k_1 = rotm2quat(R_1 @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0.,
                                                                          1.]]))  # this rotation is to set initial orientation to match with true
        z_q_k_2 = rotm2quat(R_1_2 @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0.,
                                                                            1.]]))  # this rotation is to set initial orientation to match with true
        z_q_k = z_q_k_1.copy()
        z_pi_k = z_pi_k_1.copy()
        z_p_k = z_p_k_1.copy()
        perfect_metric = False
        perfect_metric_2 = False
    else:
        z_q_k_1, _, error = rotation_association(q_kp1, R_1)
        z_q_k_2, bad_attitude_measurement_flag_2, error_2 = rotation_association(q_kp1, R_1_2)
        if quat_angle_diff(z_q_k_1, q_true[i, :]) > np.deg2rad(35):
            perfect_metric = True
        else:
            perfect_metric = False

        if quat_angle_diff(z_q_k_2, q_true[i, :]) > np.deg2rad(35):
            perfect_metric_2 = True
        else:
            perfect_metric_2 = False

    if i > 0:
        LWD = 2 * quat2rotm(q_kp1).T @ (p_kp1 - p1_kp1)
        L = LWD[0];
        W = LWD[1];
        D = LWD[2]
        predictedBbox = boundingbox.from_params(p_kp1, q_kp1, L, W, D)  # just use the predicted box instead

        # first use q from R_1 to get L,W,D
        # then use z_q_k (not perfectly aligned) to get
    associatedBbox_1, Lm, Wm, Dm = boundingbox.associated(z_q_k_1, z_pi_k_1, z_p_k_1,
                                                          R_1)  # L: along x-axis, W: along y-axis D: along z-axis
    z_p1_k_1 = associatedBbox_1[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)
    associatedBbox_2, Lm_2, Wm_2, Dm_2 = boundingbox.associated(z_q_k_2, z_pi_k_2, z_p_k_2, R_1_2)
    z_p1_k_2 = associatedBbox_2[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)

    if i == 0:
        associatedBbox = associatedBbox_1.copy()
        z_p1_k = associatedBbox_1[:, 0]
        z_q_k_1_previous = z_q_k_1.copy()
        z_q_k_2_previous = z_q_k_2.copy()

    if i > 0:
        ###########################################################################3
        ransac_pred_diff = np.rad2deg(quat_angle_diff(q_kp1, z_q_k_2))
        pca_pred_diff = np.rad2deg(quat_angle_diff(q_kp1, z_q_k_1))
        ransac_pca_diff = np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_1))
        pca_prev_diff = np.rad2deg(quat_angle_diff(z_q_k_1, z_q_k_1_previous))
        ransac_prev_diff = np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_2_previous))
        pca_true_diff = np.rad2deg(quat_angle_diff(z_q_k_1, q_true[i, :]))
        ransac_true_diff = np.rad2deg(quat_angle_diff(z_q_k_2, q_true[i, :]))
        pred_true_diff = np.rad2deg(quat_angle_diff(q_kp1, q_true[i, :]))
        # pred_prev_diff = np.rad2deg(quat_angle_diff(q_kp1, q_km1))
        ran_pred_thresh = 20
        pca_pred_thresh = 20
        ran_pca_thresh = 25
        pca_prev_thresh = 10
        ran_prev_thresh = 10
        # pred_prev_diff = 15

        if pca_true_diff < 20 or ransac_true_diff < 20:
            if pca_true_diff > ransac_true_diff:
                true_ran += 1
            else:
                true_pca += 1
        elif pca_true_diff < ransac_true_diff and pca_true_diff < pred_true_diff:
            true_pca += 1
        elif ransac_true_diff < pca_true_diff and ransac_true_diff < pred_true_diff:
            true_ran += 1
        else:
            true_pred += 1


        # super metric
    if i > settling_time + 40:
        if ransac_pred_diff > ran_pred_thresh:
            if pca_pred_diff > pca_pred_thresh:
                if ransac_pca_diff > ran_pca_thresh:
                    if ransac_prev_diff > ran_prev_thresh:
                        # ransac, prediction, pca, and changes between previous measurements are off
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred1 += 1
                            prediction += 1
                            print("using predicted 1")
                        # ransac, prediction, pca, previous ransac are off, but current and previous pca show reasonable change
                        else:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred2 += 2
                            prediction += 1
                            print("using predicted 2")
                    else:
                        # ransac, prediction, pca, previous pca are off, but current and previous ransac show reasonable change
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran8 += 1
                            ransac += 1
                            print("using ransac 8")
                        # ransac, prediction and pca are off, but both measurements show consistent change
                        else:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred4 += 1
                            prediction += 1
                            print("using predicted 4")
                else:
                    if ransac_prev_diff > ran_prev_thresh:
                        # ransac and pca are close but far from prediction, but both previous measurements are not consistent change
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred5 += 1
                            prediction += 1
                            print("using predicted 5")
                        # ransac and pca are close but far from prediction, ransac previous inconsistent but pca previous consistent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca1 += 1
                            pca += 1
                            print("using pca 1")
                    else:
                        # ransac and pca are close but far from prediction, ransac previous consistent but pca previous inconsistent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred6 += 1
                            prediction += 1
                            print("using predicted 6")
                        # ransac and pca are close but far from prediction, ransac previous consistent and pca previous consistent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca2 +=1
                            pca += 1
                            print("using pca 2")
            else:
                if ransac_pca_diff > ran_pca_thresh:
                    if ransac_prev_diff > ran_prev_thresh:
                        # pca and prediction are close, pca and ransac are far, ransac and pred are far, and both ransac and pca are inconsistent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred7 += 1
                            prediction += 1
                            print("using predicted 7")
                        # pca and prediction are close, pca and ransac are far, ransac and pred are far,  pca consistent, ransac inconsistent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca3 += 1
                            pca += 1
                            print("using pca 3")
                    else:
                        # pca and prediction are close, pca and ransac are far, ransac and pred are far, pca inconsistent, ransac consistent
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran1 += 1
                            ransac += 1
                            print("using ransac 1")
                        # pca and prediction are close, pca and ransac are far, ransac and pred are far, and both ransac and pca are consistent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca4 += 1
                            pca += 1
                            print("using pca 4")
                else:
                    if ransac_prev_diff > ran_prev_thresh:
                        # pca and prediction are close, pca and ransac are close, ransac and pred are far, and both ransac and pca are inconsistent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred8 += 1
                            prediction += 1
                            print("using predicted 8")
                        # pca and prediction are close, pca and ransac are close, ransac and pred are far, and ransac inconsistent, pca consistent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca5 += 1
                            pca += 1
                            print("using pca 5")
                    else:
                        # pca and prediction are close, pca and ransac are close, ransac and pred are far, and ransac consistent, pca inconsistent
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran2 += 1
                            ransac += 1
                            print("using ransac 2")
                        # pca and prediction are close, pca and ransac are close, ransac and pred are far, pca and ransac consistent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca6 += 1
                            pca += 1
                            print("using pca 6")
        else:
            if pca_pred_diff > pca_pred_thresh:
                if ransac_pca_diff > ran_pca_thresh:
                    if ransac_prev_diff > ran_prev_thresh:
                        # ransac pred close, pca pred far, ransac pca far, ransac and pca inconsistent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred9 += 1
                            prediction += 1
                            print("using predicted 9")
                        # ransac pred close, pca pred far, ransac pca far, ransac inconsisent and pca consistent
                        else:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred10 += 1
                            prediction += 1
                            print("using predicted 10")
                    else:
                        # ransac pred close, pca pred far, ransac pca far, ransac consisent and pca inconsistent
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran3 += 1
                            ransac += 1
                            print("using ransac 3")
                        # ransac pred close, pca pred far, ransac pca far, ransac and pca consisent
                        else:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran4 += 1
                            ransac += 1
                            print("using ransac 4")
                else:
                    if ransac_prev_diff > ran_prev_thresh:
                        # ransac pred close, pca pred far, ransac pca close, ransac and pca inconsisent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred11 += 1
                            prediction += 1
                            print("using predicted 11")
                        # ransac pred close, pca pred far, ransac pca close, ransac inconsistent and pca consisent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca7 += 1
                            pca += 1
                            print("using pca 7")
                    else:
                        # ransac pred close, pca pred far, ransac pca close, ransac consistent and pca inconsisent
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            ran5 += 1
                            adapt = False
                            ransac += 1
                            print("using ransac 5")
                        # ransac pred close, pca pred far, ransac pca close, ransac consistent and pca consisent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca8 += 1
                            pca += 1
                            print("using pca 8")
            else:
                if ransac_pca_diff > ran_pca_thresh:
                    if ransac_prev_diff > ran_prev_thresh:
                        # ransac pred close, pca pred close, ransac pca far, ransac inconsistent and pca inconsisent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred12 += 1
                            prediction += 1
                            print("using predicted 12")
                        # ransac pred close, pca pred close, ransac pca far, ransac inconsistent and pca consisent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca9 += 1
                            pca += 1
                            print("using pca 9")
                    else:
                        # ransac pred close, pca pred close, ransac pca far, ransac consistent and pca inconsisent
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran6 += 1
                            ransac += 1
                            print("using ransac 6")
                        # ransac pred close, pca pred close, ransac pca far, ransac consistent and pca consisent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca10 += 1
                            pca += 1
                            print("using pca 10")
                else:
                    if ransac_prev_diff > ran_prev_thresh:
                        # ransac pred close, pca pred close, ransac pca close, ransac inconsistent and pca inconsisent
                        if pca_prev_diff > pca_prev_thresh:
                            # trust prediction
                            associatedBbox = predictedBbox.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox[:, 0]
                            adapt = True
                            pred13 += 1
                            prediction += 1
                            print("using predicted 13")
                        # ransac pred close, pca pred close, ransac pca close, ransac inconsistent and pca consisent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            pca11 += 1
                            pca += 1
                            print("using pca 11")
                    else:
                        # ransac pred close, pca pred close, ransac pca close, ransac consistent and pca inconsisent
                        if pca_prev_diff > pca_prev_thresh:
                            z_q_k = z_q_k_2.copy()
                            z_pi_k = z_pi_k_2.copy()
                            z_p_k = z_p_k_2.copy()
                            z_p1_k = associatedBbox_2[:, 0]
                            associatedBbox = associatedBbox_2.copy()
                            adapt = False
                            ran7 += 1
                            ransac += 1
                            print("using ransac 7")
                        # ransac pred close, pca pred close, ransac pca close, ransac consistent and pca consisent
                        else:
                            z_q_k = z_q_k_1.copy()
                            z_pi_k = z_pi_k_1.copy()
                            z_p_k = z_p_k_1.copy()
                            z_p1_k = associatedBbox_1[:, 0]
                            associatedBbox = associatedBbox_1.copy()
                            adapt = False
                            print("using pca 12")
                            pca12 += 1
                            pca += 1

    elif i == 0:
        z_q_k = z_q_k_1.copy()
        z_pi_k = z_pi_k_1.copy()
        z_p_k = z_p_k_1.copy()
        z_p1_k = associatedBbox_1[:, 0]
        associatedBbox = associatedBbox_1.copy()
        adapt = False
        print("using pca 14")
    else:
        if ransac_pred_diff > pca_pred_diff:
            z_q_k = z_q_k_1.copy()
            z_pi_k = z_pi_k_1.copy()
            z_p_k = z_p_k_1.copy()
            z_p1_k = associatedBbox_1[:, 0]
            associatedBbox = associatedBbox_1.copy()
            adapt = False
            print("using pca 13")
        else:
            z_q_k = z_q_k_2.copy()
            z_pi_k = z_pi_k_2.copy()
            z_p_k = z_p_k_2.copy()
            z_p1_k = associatedBbox_2[:, 0]
            associatedBbox = associatedBbox_2.copy()
            adapt = False
            print("using ransac 9")

        ######################################

    without_correction.append(z_p_k)
    bbox1_dimensions.append([Lm, Wm, Dm])
    bbox2_dimensions.append([Lm_2, Wm_2, Dm_2])
    if curr_t >= (t_start + t_interval):
        z_p_k_z = correct_bias(z_p_k, i, dt, parameters, constants, Rot_L_to_B[i], Rot_B_to_L[i])
        z_p_k = z_p_k_z

    # if np.rad2deg(quat_angle_diff(z_q_k_1, q_true[i, :])) - np.rad2deg(quat_angle_diff(z_q_k_2, q_true[i, :])) > 25 or perfect_metric == True or np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_1)) > 30:
    #     visualize_flag = True
    # if perfect_metric == True:# or perfect_metric_2 == True:
    #     visualize_flag = True
    # visualize_flag = False
    if visualize_flag:
        # if False:
        print('PCA True diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_1, q_true[i, :]))))
        print('Ransac True diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2, q_true[i, :]))))
        print('Pred True diff.:' + str(np.rad2deg(quat_angle_diff(q_kp1, q_true[i, :]))))
        print('PCA Pred diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_1, q_kp1))))
        print('Ransac Pred diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2, q_kp1))))
        print('Ransac PCA diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_1))))
        print('PCA Prev. diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_1_previous, z_q_k_1))))
        print('Ransac Prev. diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2_previous, z_q_k_2))))
        print(perfect_metric)
        # _ = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, True, visualize_flag)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.legend()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.title.set_text(
            f'Time={i * dt}s' + '\n' + f'Pred. Length={round(L, 2)}m ' + f'Width={round(W, 2)}m ' + f'Height={round(D, 2)}m' + '\n' + f'Meas. Length={round(Lm, 2)}m ' + f'Width={round(Wm, 2)}m ' + f'Height={round(Dm, 2)}m')
        # width = orange to green, blue to green
        # length = orange to cyan, blue to cyan
        # height = orange to magenta, blue to magenta
        ax.scatter(X_i, Y_i, Z_i, color='black', marker='o', s=2)
        # ax.scatter(p1_kp1[0], p1_kp1[1], p1_kp1[2], marker='o', color='r')


        # print(x_kp1)
        # drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'orange', 1)
        drawrectangle(ax, associatedBbox_1[:, 0], associatedBbox_1[:, 1], associatedBbox_1[:, 2],
                      associatedBbox_1[:, 3],
                      associatedBbox_1[:, 4], associatedBbox_1[:, 5], associatedBbox_1[:, 6], associatedBbox_1[:, 7],
                      'b', 2, 'PCA')

        drawrectangle(ax, associatedBbox_2[:, 0], associatedBbox_2[:, 1], associatedBbox_2[:, 2],
                      associatedBbox_2[:, 3],
                      associatedBbox_2[:, 4], associatedBbox_2[:, 5], associatedBbox_2[:, 6], associatedBbox_2[:, 7],
                      'orange', 2, 'RANSAC')

        # drawrectangle(ax, associatedBbox[:, 0], associatedBbox[:, 1], associatedBbox[:, 2], associatedBbox[:, 3],
        #           associatedBbox[:, 4], associatedBbox[:, 5], associatedBbox[:, 6], associatedBbox[:, 7], 'orange', 2)

        # drawrectangle(ax, z_pi_k[:, 0], z_pi_k[:, 1], z_pi_k[:, 2], z_pi_k[:, 3],
        #               z_pi_k[:, 4], z_pi_k[:, 5], z_pi_k[:, 6], z_pi_k[:, 7], 'r', 1)
        # ax.scatter(p1_kp1[0], p1_kp1[1], p1_kp1[2], color='b', s=20)
        # drawrectangle(ax, predictedBbox[:, 0], predictedBbox[:, 1], predictedBbox[:, 2], predictedBbox[:, 3],
        #               predictedBbox[:, 4], predictedBbox[:, 5], predictedBbox[:, 6], predictedBbox[:, 7], 'r', 1)

        # ax.scatter(predictedBbox[0, 0], predictedBbox[1, 0], predictedBbox[2, 0], color='orange', label='Vertex 1 Pred.')
        # ax.scatter(associatedBbox[0, 0], associatedBbox[1, 0], associatedBbox[2, 0], color='blue',
        #            label='Vertex 1 Meas.')


        Rot_measured = quat2rotm(z_q_k_1)

        Rot_measured_2 = quat2rotm(z_q_k_2)
        # Rot_measured_2 = R_1_2
        # normal_vecs = normal_vecs.T

        R_estimated = quat2rotm(q_kp1)

        R_true = quat2rotm(q_true[i, :])

        # plot measured
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 0]],
                [z_p_k[2], z_p_k[2] + Rot_measured[2, 0]],
                color='blue', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 1]],
                [z_p_k[2], z_p_k[2] + Rot_measured[2, 1]],
                color='blue', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 2]],
                [z_p_k[2], z_p_k[2] + Rot_measured[2, 2]],
                color='b', linewidth=4)

        # plot measured
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 0]],
                [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 0]],
                color='orange', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 1]],
                [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 1]],
                color='orange', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 2]],
                [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 2]],
                color='orange', linewidth=4)
        #
        # Rot_measured_2 = R_1_2
        # plot measured
        # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 0]],
        #         [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 0]],
        #         color='red', linewidth=4)
        # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 1]],
        #         [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 1]],
        #         color='red', linewidth=4)
        # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 2]],
        #         [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 2]],
        #         color='red', linewidth=4)

        # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 0]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 0]],
        #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 0]],
        #         color='blue', linewidth=4)
        # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 1]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 1]],
        #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 1]],
        #         color='blue', linewidth=4)
        # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 2]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 2]],
        #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 2]],
        #         color='blue', linewidth=4)
        #

        # plot current estimate of ekf
        ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 0]], [z_p_k[1], z_p_k[1] + R_estimated[1, 0]],
                [z_p_k[2], z_p_k[2] + R_estimated[2, 0]],
                color='red', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 1]], [z_p_k[1], z_p_k[1] + R_estimated[1, 1]],
                [z_p_k[2], z_p_k[2] + R_estimated[2, 1]],
                color='red', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 2]], [z_p_k[1], z_p_k[1] + R_estimated[1, 2]],
                [z_p_k[2], z_p_k[2] + R_estimated[2, 2]],
                color='red', linewidth=4, label='Predicted')

        # plot true
        ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 0]], [z_p_k[1], z_p_k[1] + R_true[1, 0]],
                [z_p_k[2], z_p_k[2] + R_true[2, 0]],
                color='green', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 1]], [z_p_k[1], z_p_k[1] + R_true[1, 1]],
                [z_p_k[2], z_p_k[2] + R_true[2, 1]],
                color='green', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 2]], [z_p_k[1], z_p_k[1] + R_true[1, 2]],
                [z_p_k[2], z_p_k[2] + R_true[2, 2]],
                color='green', linewidth=4, label='True')

        # plot b_frame
        # ax.plot([0., 0. + Rot_B_to_L[i][0, 0]], [0., 0. + Rot_B_to_L[i][1, 0]],
        #         [0., 0. + Rot_B_to_L[i][2, 0]],
        #         color='r', linewidth=1)
        # ax.plot([0., 0. + Rot_B_to_L[i][0, 1]], [0., 0. + Rot_B_to_L[i][1, 1]],
        #         [0., 0. + Rot_B_to_L[i][2, 1]],
        #         color='g', linewidth=1)
        # ax.plot([0., 0. + Rot_B_to_L[i][0, 2]], [0., 0. + Rot_B_to_L[i][1, 2]],
        #         [0., 0. + Rot_B_to_L[i][2, 2]],
        #         color='b', linewidth=1)

        # black is axis of rotation
        # ax.plot([z_p_k[0], z_p_k[0] + 1], [z_p_k[1], z_p_k[1] + 1],
        #         [z_p_k[2], z_p_k[2] + 1],
        #         color='black', linewidth=4)

        # outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # Visualize the inliers (plane) and outliers
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red plane
        # outlier_cloud.paint_uniform_color([0.0, 1, 0])  # Green remaining points
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        # ax.scatter(x_k[0], x_k[1], x_k[2], color='orange' )
        # ax.scatter(z_p_k_1[0], z_p_k_1[1], z_p_k_1[2], color='b', label='Box Centroid')
        # ax.scatter(debris_pos[i,0], debris_pos[i,1], debris_pos[i,2], color='g', label='True Position')
        ax.legend()
        ax.set_aspect('equal', 'box')
        plt.show()



    # find angular velocity from LOS velocities
    if i > 0:
        # 1. Linear Least Squares
        omega_LLS_B = estimate_LLS(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i] @ z_p_k, Rot_L_to_B[i] @ v_k, VBs[i])
        omega_LLS = Rot_B_to_L[i] @ omega_LLS_B

    # 2. Rotation of B Frame
    omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)

    # 3. Kabsch
    ################ to use Kabsch you need i > 0, to wait for state initializations?
    if i == 0:
        omega_los_L = np.array([0, 0, 0])
        prev_box_L = np.transpose(copy.deepcopy(associatedBbox_1))
        prev_box_B = (Rot_L_to_B[i] @ prev_box_L.T).T
    else:
        cur_box_L = np.transpose(copy.deepcopy(associatedBbox_1))
        cur_box_B = (Rot_L_to_B[i] @ cur_box_L.T).T
        # rotate previous box with everything else
        # prev_box_B = (rodrigues((omega_LLS + omega_L_to_B), dt) @ prev_box_B.T).T
        omega_los_B = estimate_kabsch(prev_box_B, cur_box_B, dt)
        prev_box_B = cur_box_B.copy()  # for next iteration

        # using moving average to smooth out omega_los_B
        omega_kabsch_b_box[i % n_moving_average] = omega_los_B
        if i < n_moving_average:
            omega_los_B_averaged = np.mean(omega_kabsch_b_box[0:i + 1], axis=0)
        else:
            omega_los_B_averaged = np.mean(omega_kabsch_b_box, axis=0)
        omega_los_L = Rot_B_to_L[i] @ omega_los_B_averaged

    # 3b Kabsch alternate
    # this should be done in the B frame
    # omega_los_Ls = np.zeros([n_moving_average, 3])
    # if i==0:
    #     omega_los_L = np.array([0,0,0])
    # else:
    #     prev_q = z_s[i-1][9:]
    #     cur_q = z_q_k
    #     prev_R_in_B = Rot_L_to_B[i] @ quat2rotm(prev_q)
    #     cur_R_in_B = Rot_L_to_B[i] @ quat2rotm(cur_q)
    #     prev_R_in_B_after_LLS = prev_R_in_B @ rodrigues(omega_LLS_B, dt)
    #     net_R_in_B = (prev_R_in_B_after_LLS.T) @ (cur_R_in_B)
    #     rotation_axis_bbox, rotation_angle_bbox = R_to_axis_angle(net_R_in_B)
    #     omega_bbox = rotation_axis_bbox * rotation_angle_bbox / dt
    #     # take only the z_component (along LOS) of omega_bbox
    #     omega_los_B_q = np.array([0,0,omega_bbox[2]])
    #     omega_los_L_q = (Rot_B_to_L[i] @ omega_los_B_q.reshape([3,1])).reshape(3)
    #     omega_los_Ls[i%n_moving_average] = omega_los_L_q
    #     if i < n_moving_average:
    #         omega_los_L_averaged = np.mean(omega_los_Ls[0:i+1], axis=0)
    #     else:
    #         omega_los_L_averaged = np.mean(omega_los_Ls, axis=0)

    # Combine angular velocity estimates
    if i == 0:
        z_omega_k = omega_0
    elif i <= settling_time:
        z_omega_k = omega_LLS + omega_L_to_B  # ignores kabsch
    else:
        z_omega_k = omega_LLS + omega_L_to_B + omega_los_L

    #################################

    # Compute Measurement Vector
    z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_q_k])

    ##############
    # Update - Combine Measurement and Estimates
    ##############

    # Compute Measurement Vector
    # if False:
    if adapt:
        z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k])
        H = H2
        R = R2
    else:
        z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_q_k])
        H = H1
        R = R1

    # Set initial states to measurements
    if i == 0:
        x_k = np.hstack([np.array([-170., -350., -20.]), vT_0, z_omega_k, z_p1_k, q_ini])  # state
        P_k = P_0.copy()  # covariance matrix

    num_meas = len(z_kp1)

    if i > 0:
        #################
        # iterated measurement update
        #################
        x_op = x_kp1.copy()
        P_op = P_kp1.copy()

        current_difference = 1  # initialize to a high value so that it can enter the loop, this is the current difference between states of consecutive iterations

        # iterate to desired threshold
        while current_difference > tolerance:
            ####################
            # measurement update
            ###################

            # state vector as mean for sigmapoint transform
            mu_sp_m = x_op.copy()

            # stack covariance matrix with process noise
            sigma_zz_m = P_op.copy()

            # cholesky, ensure positive definiteness
            try:
                L_m = scipy.linalg.cholesky(sigma_zz_m, lower=True)
            except np.linalg.LinAlgError:
                np.fill_diagonal(sigma_zz_m, sigma_zz_m.diagonal() + epsilon)
                L_m = scipy.linalg.cholesky(sigma_zz_m, lower=True)

            # initial sigmapoint
            sp_0_m = mu_sp_m

            # other sigmapoints
            sp_s_m = [sp_0_m]
            sqrt_term_m = np.sqrt(dimL + lambd)
            for idx in range(0, dimL):
                col_i_L_m = L_m[:, idx]
                sp_i_m = mu_sp_m + sqrt_term_m * col_i_L_m
                sp_s_m.append(sp_i_m)
            for idx in range(0, dimL):
                col_i_L_m = L_m[:, idx]
                sp_i_L_m = mu_sp_m - sqrt_term_m * col_i_L_m
                sp_s_m.append(sp_i_L_m)

            # pass each point through measurement model
            y_kp1_s_m = []

            mu_y_kp1_m = np.zeros((num_meas,))
            for jdx, sp_m in enumerate(sp_s_m):
                y_kp1_m = H @ sp_m
                if jdx == 0:
                    mu_y_kp1_m += w_0_m * y_kp1_m
                else:
                    mu_y_kp1_m += w_j_m * y_kp1_m
                y_kp1_s_m.append(y_kp1_m)

            # various aposteriori covariances
            sigma_yy = np.zeros((num_meas, num_meas))
            sigma_xy = np.zeros((num_states, num_meas))
            for kdx, sp in enumerate(sp_s_m):
                diff_x_m = sp - x_kp1
                diff_y_m = y_kp1_s_m[kdx] - mu_y_kp1_m
                if kdx == 0:
                    sigma_yy += w_0_c * np.outer(diff_y_m, diff_y_m.T)
                    sigma_xy += w_0_c * np.outer(diff_x_m, diff_y_m.T)
                else:
                    sigma_yy += w_j_c * np.outer(diff_y_m, diff_y_m.T)
                    sigma_xy += w_j_c * np.outer(diff_x_m, diff_y_m.T)

            # Kalman gain
            sigma_yy += R
            K_kp1 = np.matmul(sigma_xy, np.linalg.inv(sigma_yy))

            # Calculate Residual
            res_kp1 = z_kp1 - mu_y_kp1_m

            x_op_prev = x_op.copy()

            # Update State
            x_op = x_op + K_kp1 @ res_kp1

            # Update Covariance
            P_op = P_op - K_kp1 @ sigma_yy @ K_kp1.T

            current_difference = np.linalg.norm(x_op - x_op_prev)

        # Transfer states and covariance from kp1 to k
        P_k = P_op.copy()
        x_op[12:] = normalize_quat(x_op[12:])
        x_k = x_op.copy()

        x_p_k = x_k[0:3]
        x_p1_k = x_k[9:12]
        x_q_k = x_k[12:16]
        rotation_errors.append(quat_angle_diff(x_q_k, q_true[i, :]))
        Le, We, De = get_dimensions(x_p1_k, x_p_k, x_q_k)
        bbox3_dimensions.append([Le, We, De])

        # smooth out covariance off diagonals
        P_k = 0.5 * P_k + 0.5 * P_k.T

        z_q_k_1_previous = z_q_k_1.copy()
        z_q_k_2_previous = z_q_k_2.copy()

    z_s.append(z_kp1)
    z_pcas.append(np.hstack([z_p_k_1, z_omega_k, associatedBbox_1[:, 0], z_q_k_1]))
    z_rans.append(np.hstack([z_p_k_2, z_omega_k, associatedBbox_2[:, 0], z_q_k_2]))

    # Append for analysis
    P_s.append(P_k)
    x_s.append(x_k)
    estimated_pos.append(x_k[:3])

##############
# Plot relevant figures
##############

print("True Pred: " + str(true_pred))
print("True ran: " + str(true_ran))
print("True pca: " + str(true_pca))
print("Used Pred: " + str(prediction))
print("Used Ran: " + str(ransac))
print("USEd PCA: " + str(pca))
print("PCA bins: " + str([pca1, pca2, pca3, pca4, pca5, pca6, pca7, pca8, pca9, pca10, pca11, pca12]))
print("Pred bins: " + str([pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12, pred13]))
print("Ran bins: " + str([ran1, ran2, ran3, ran4, ran5, ran6, ran7, ran8]))






m1 = len(x_s)

z_s = padding_nan(z_s)
x_s = np.array(x_s)
x_s = x_s[1:, :]
q_true = np.array(q_true)
original_pos_meas = np.array(original_pos_meas)
centroids_inB = np.array(centroids_inB)
true_pos_inB = np.array(true_pos_inB)
z_pcas = np.array(z_pcas)
z_rans = np.array(z_rans)
without_correction = np.array(without_correction)
bbox1_dimensions = np.array(bbox1_dimensions)
bbox2_dimensions = np.array(bbox2_dimensions)
bbox3_dimensions = np.array(bbox3_dimensions)


#####################
# errors over last 50 seconds
####################

# position
start_time_2 = -int(50 / dt)
rmse_px = np.sqrt(np.mean((x_s[start_time_2:, 0] - debris_pos[start_time_2:nframes, 0]) ** 2))
rmse_py = np.sqrt(np.mean((x_s[start_time_2:, 1] - debris_pos[start_time_2:nframes, 1]) ** 2))
rmse_pz = np.sqrt(np.mean((x_s[start_time_2:, 2] - debris_pos[start_time_2:nframes, 2]) ** 2))
me_px = np.mean(x_s[start_time_2:, 0] - debris_pos[start_time_2:nframes, 0])
me_py = np.mean(x_s[start_time_2:, 1] - debris_pos[start_time_2:nframes, 1])
me_pz = np.mean(x_s[start_time_2:, 2] - debris_pos[start_time_2:nframes, 2])

# angular velocity
rmse_omx = np.sqrt(np.mean((x_s[start_time_2:, 6] - omega_true[0]) ** 2))
rmse_omy = np.sqrt(np.mean((x_s[start_time_2:, 7] - omega_true[1]) ** 2))
rmse_omz = np.sqrt(np.mean((x_s[start_time_2:, 8] - omega_true[2]) ** 2))
me_omx = np.mean((x_s[start_time_2:, 6] - omega_true[0]))
me_omy = np.mean((x_s[start_time_2:, 7] - omega_true[1]))
me_omz = np.mean((x_s[start_time_2:, 8] - omega_true[2]))

# linear velocity
rmse_vdx = np.sqrt(np.mean((x_s[start_time_2:, 3] - debris_vel[start_time_2:nframes, 0]) ** 2))
rmse_vdy = np.sqrt(np.mean((x_s[start_time_2:, 4] - debris_vel[start_time_2:nframes, 1]) ** 2))
rmse_vdz = np.sqrt(np.mean((x_s[start_time_2:, 5] - debris_vel[start_time_2:nframes, 2]) ** 2))
me_vdx = np.mean(x_s[start_time_2:, 3] - debris_vel[start_time_2:nframes, 0])
me_vdy = np.mean(x_s[start_time_2:, 4] - debris_vel[start_time_2:nframes, 1])
me_vdz = np.mean(x_s[start_time_2:, 5] - debris_vel[start_time_2:nframes, 2])

# orientation rmse
rmse_q = np.sqrt(np.mean(np.rad2deg(rotation_errors[start_time_2:nframes]) ** 2))

# bias errors
b_start = int(t_start / dt)
b_end = int((t_start + t_interval) / dt)
rmse_x_before = np.sqrt(np.mean((x_s[b_start:b_end, 0] - debris_pos[b_start:b_end, 0]) ** 2))
rmse_y_before = np.sqrt(np.mean((x_s[b_start:b_end, 1] - debris_pos[b_start:b_end, 1]) ** 2))
rmse_z_before = np.sqrt(np.mean((x_s[b_start:b_end, 2] - debris_pos[b_start:b_end, 2]) ** 2))
me_x_before = np.mean(x_s[b_start:b_end, 0] - debris_pos[b_start:b_end, 0])
me_y_before = np.mean(x_s[b_start:b_end, 1] - debris_pos[b_start:b_end, 1])
me_z_before = np.mean(x_s[b_start:b_end, 2] - debris_pos[b_start:b_end, 2])

rmse_x_after = np.sqrt(np.mean((x_s[b_end:, 0] - debris_pos[b_end:nframes, 0]) ** 2))
rmse_y_after = np.sqrt(np.mean((x_s[b_end:, 1] - debris_pos[b_end:nframes, 1]) ** 2))
rmse_z_after = np.sqrt(np.mean((x_s[b_end:, 2] - debris_pos[b_end:nframes, 2]) ** 2))
me_x_after = np.mean(x_s[b_end:, 0] - debris_pos[b_end:nframes, 0])
me_y_after = np.mean(x_s[b_end:, 1] - debris_pos[b_end:nframes, 1])
me_z_after = np.mean(x_s[b_end:, 2] - debris_pos[b_end:nframes, 2])

print("Position RMSE: " + str([rmse_px, rmse_py, rmse_pz]))
print("Position ME: " + str([me_px, me_py, me_pz]))
print("Ang. Vel. RMSE: " + str([rmse_omx, rmse_omy, rmse_omz]))
print("Ang. Vel. ME: " + str([me_omx, me_omy, me_omz]))
print("Lin. Vel. RMSE: " + str([rmse_vdx, rmse_vdy, rmse_vdz]))
print("Lin. ME: " + str([me_vdx, me_vdy, me_vdz]))
print("Bias before RMSE: " + str([rmse_x_before, rmse_y_before, rmse_z_before]))
print("Bias after RMSE: " + str([rmse_x_after, rmse_y_after, rmse_z_after]))
print("Bias before ME: " + str([me_x_before, me_y_before, me_z_before]))
print("Bias before ME: " + str([me_x_after, me_y_after, me_z_after]))
print("Orientation RMSE: " + str(rmse_q))



plt.rcParams.update({'font.size': 12})
plt.rcParams['text.usetex'] = True

# fig = plt.figure()
# plt.plot(np.arange(0, dt*nframes, dt), centroids_inB[:, 0] - true_pos_inB[:, 0])
# plt.xlabel('Time (s)')
# plt.ylabel('$\displaystyle p_x$ (m)')
#
# fig = plt.figure()
# plt.plot(np.arange(0, dt*nframes, dt), centroids_inB[:, 1] - true_pos_inB[:, 1])
# plt.xlabel('Time (s)')
# plt.ylabel('$\displaystyle p_y$ (m)')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), centroids_inB[:, 2] - true_pos_inB[:, 2], label='Computed', color='blue')
plt.plot(np.arange(0, dt*nframes, dt), np.zeros_like(true_pos_inB), label='True', color='Green', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_z$ (m)')
plt.legend()

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:,0] - debris_pos[:,0], label='Computed', linewidth=1, color='blue')
plt.plot(np.arange(0, dt*nframes, dt), without_correction[:, 0] - debris_pos[:,0], label='Original', linewidth=1, color='brown')
plt.plot(np.arange(0, dt*nframes, dt), np.zeros_like(z_s[:, 0]), label='True', color='green', linestyle='--', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_x$ (m)')
plt.legend()
#plt.title('X Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:,1] - debris_pos[:,1], label='Computed', linewidth=1, color='blue')
plt.plot(np.arange(0, dt*nframes, dt), without_correction[:, 1] - debris_pos[:,1], label='Original', linewidth=1, color='brown')
plt.plot(np.arange(0, dt*nframes, dt), np.zeros_like(z_s[:, 1]), label='True', color='green', linestyle='--', linewidth=1)
# plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,1] - debris_pos[:,1], label='Estimated', linewidth=2)
# plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,1], label='True', linewidth=1, linestyle='dashed')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_y$ (m)')
#plt.title('Y Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:,2] - debris_pos[:,2], label='Computed', linewidth=1, color='blue')
plt.plot(np.arange(0, dt*nframes, dt), without_correction[:, 2] - debris_pos[:,2], label='Original', linewidth=1, color='brown')
plt.plot(np.arange(0, dt*nframes, dt), np.zeros_like(z_s[:, 2]), label='True', color='green', linestyle='--', linewidth=1)

# plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,2], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_z$ (m)')
#plt.title('Z Position')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.scatter(debris_pos[1, 0], debris_pos[1, 1], debris_pos[1, 2], color='orange', marker='o', s=20)
ax.scatter(debris_pos[-1, 0], debris_pos[-1, 1], debris_pos[-1, 2], color='k', marker='o', s=20)
ax.scatter(z_s[:, 0], z_s[:, 1], z_s[:, 2], color='b', s=0.3, linewidths=0)
ax.plot(debris_pos[:, 0], debris_pos[:, 1], debris_pos[:, 2], color='g')
ax.legend(['Start', 'End', 'Computed Centroid Positions', 'True Centroid Positions'])
# plt.xlim([-170.5, -167.5])
# plt.ylim([-351, -306])
# ax.set_zlim(-20, -9)

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 3], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 6], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), omega_true[0] * np.ones([nframes, 1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle \Omega_x$ (rad/s)')
# plt.title('$\displaystyle\Omega_x$')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 4], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 7], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), omega_true[1] * np.ones([nframes, 1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle \Omega_y$ (rad/s)')
# plt.title('Omega Y')


fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 5], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 8], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), omega_true[2] * np.ones([nframes, 1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()

plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle \Omega_z$ (rad/s)')
# plt.title('Omega Z')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 6] - omega_true[0], label='Error $\displaystyle \Omega_x$', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 7] - omega_true[1], label='Error $\displaystyle \Omega_y$', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 8] - omega_true[2], label='Error $\displaystyle \Omega_z$', linewidth=2)
# plt.plot(np.arange(0, dt*nframes, dt), np.zeros([nframes,1]), linewidth = 1) # draw line at zero
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity Error (rad/s)')
# plt.title('Angular Velocity Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 0] - debris_pos[:nframes, 0], label='Error $\displaystyle p_x$',
         linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 1] - debris_pos[:nframes, 1], label='Error $\displaystyle p_y$',
         linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 2] - debris_pos[:nframes, 2], label='Error $\displaystyle p_z$',
         linewidth=2)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
# plt.title('Position Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 0], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 0], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), debris_pos[:nframes, 0], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_x$ (m)')
# plt.title('X Position')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 1], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 1], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), debris_pos[:nframes, 1], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_y$ (m)')
# plt.title('Y Position')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 2], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 2], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), debris_pos[:nframes, 2], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_z$ (m)')
# plt.title('Z Position')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 3] - debris_vel[:nframes, 0], label='Error $\displaystyle v_{Dx}$',
         linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 4] - debris_vel[:nframes, 1], label='Error $\displaystyle v_{Dy}$',
         linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 5] - debris_vel[:nframes, 2], label='Error $\displaystyle v_{Dz}$',
         linewidth=1)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity Error (m/s)')
# plt.title('Velocity Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 3], label='Estimated', color='Orange', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), debris_vel[:nframes, 0], label='True', color='green', linewidth=1, linestyle='--')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Dx}$ (m/s)')
# plt.title('Velocity in X')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 4], label='Estimated', color='Orange', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), debris_vel[:nframes, 1], label='True', color='green', linewidth=1, linestyle='--')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Dy}$ (m/s)')
# plt.title('Velocity in y')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 5], label='Estimated', color='Orange', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), debris_vel[:nframes, 2], label='True', color='green', linewidth=1, linestyle='--')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Dz}$ (m/s)')
# plt.title('Velocity in z')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 9], label='$\displaystyle p_{1x}$')
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 10], label='$\displaystyle p_{1y}$')
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 11], label='$\displaystyle p_{1z}$')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Vertex $\displaystyle p_{1}$ Position (m)')
# plt.title('Position of Vertice P1 overt time')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 9], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 12], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 0], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_w$')
# plt.title('Orientation $\displaystyle q_0$')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 10], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 13], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 1], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_x$')
# plt.title('Orientation $\displaystyle q_1$')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 11], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 14], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 2], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_y$')
# plt.title('Orientation $\displaystyle q_2$')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 12], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 15], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 3], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_z$')
# plt.title('Orientation $\displaystyle q_3$')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), bbox1_dimensions[:, 0], label='Length')
plt.plot(np.arange(0, dt * nframes, dt), bbox1_dimensions[:, 1], label='Width')
plt.plot(np.arange(0, dt * nframes, dt), bbox1_dimensions[:, 2], label='Height')
plt.legend()
plt.title('PCA Box Dimensions')
plt.xlabel('Time (s)')
plt.ylabel('Size (m)')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), bbox2_dimensions[:, 0], label='Length')
plt.plot(np.arange(0, dt * nframes, dt), bbox2_dimensions[:, 1], label='Width')
plt.plot(np.arange(0, dt * nframes, dt), bbox2_dimensions[:, 2], label='Height')
plt.legend()
plt.title('RANSAC Box Dimensions')
plt.xlabel('Time (s)')
plt.ylabel('Size (m)')

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), bbox3_dimensions[:, 0], label='Length')
plt.plot(np.arange(0, dt * nframes, dt), bbox3_dimensions[:, 1], label='Width')
plt.plot(np.arange(0, dt * nframes, dt), bbox3_dimensions[:, 2], label='Height')
plt.legend()
plt.title('Filtered Box Dimensions')
plt.xlabel('Time (s)')
plt.ylabel('Size (m)')
"""
fig = plt.figure()
true_b = []
for i in range(nframes):
    true_b.append(Rot_L_to_B[i] @ [1,1,1])
true_b = np.array(true_b)
plt.plot(np.arange(0, dt*nframes, dt), true_b[:,2], label='True')

plt.plot(np.arange(0, dt*nframes, dt), omega_kabsch_b[:,2], label='Computed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('$\displaystyle {}^B \Omega_z$ from Kabsch')
"""

plt.show()
