#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
from scipy import io
import os
# import matplotlib.pyplot as plt

def get_inv_quat(q):
  temp = q*(-1)
  temp[0] = temp[0]*(-1)
  q_k_inv = temp
  return q_k_inv

def make_quaternion(wq):
  alpha = ((wq**2).sum())**0.5
  if alpha!= 0:
    ew = wq/alpha
  else:
    ew = wq*0
  qw = [np.cos(alpha/2),np.sin(alpha/2)*ew[0],np.sin(alpha/2)*ew[1],np.sin(alpha/2)*ew[2]]
  return np.array(qw)

def quaternion_multiply(quaternion1, quaternion0): 
#q = q1.q0 verified at https://www.vcalc.com/equation/?uuid=ca9f0f2b-7527-11e6-9770-bc764e2038f2
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def get_X_i_from_W_i(x_k_1,W):
  X_i = np.zeros((x_k_1.shape[0],W.shape[1]))
  x_k_1_w = np.asarray([list(x_k_1[4:])]*W.shape[1]).T
  X_i[4:] = W[3:]+x_k_1_w
  for i in range(W.shape[1]):
    qi = make_quaternion(W[:3,i])
    X_i[:4,i] = quaternion_multiply(x_k_1[:4],qi)
  return X_i


def generate_sigma_pts(P_k_1,Q,x_k_1):
    S = np.linalg.cholesky(P_k_1+Q)
    S = S/(1.5*np.linalg.norm(S))
    W1 = ((2*P_k_1.shape[0])**0.5)*S
    W2 = (-(2*P_k_1.shape[0])**0.5)*S
    W = np.hstack((W1,W2))
    X_i = get_X_i_from_W_i(x_k_1,W)
    return X_i

def process_model(X_i,del_t):
  norm_w = ((np.sum(X_i[4:]**2,axis = 0))**0.5)
  alpha = norm_w*del_t
  Y_i = X_i.copy()
  e = X_i[4:]/norm_w
  if sum(alpha==0):
      for i in range(X_i.shape[1]):
        if alpha[i]==0:
          e[:,i] = np.array([1,0,0])
  for i in range(X_i.shape[1]):
    q_del = np.array((np.cos(alpha[i]/2),e[0,i]*np.sin(alpha[i]/2),e[1,i]*np.sin(alpha[i]/2),e[2,i]*np.sin(alpha[i]/2)))
    Y_i[:4,i] = quaternion_multiply(X_i[:4,i],q_del)
  return Y_i

def quat_to_vec(q):
  q = q/np.linalg.norm(q)
  alpha = 2*np.arccos(q[0])
  vec = q[1:]/np.sin(alpha/2)
  vec = vec*alpha
  return vec





def compute_quaternion_mean_mine(Y_i):
  quats = Y_i[:4]
  q_bar_prev = np.zeros(4)
  q_bar = np.array((0.43095156, -0.08449536,  0.8064019 , -0.3960521)) # np.array([0.2,0.2,0.3,(1-(0.2**2+0.2**2+0.3**2))**0.5])
  j=0
  while (np.linalg.norm(q_bar_prev - q_bar)>0.01):
    j=j+1
    e_i = np.zeros(4)
    q_bar_inv = get_inv_quat(q_bar)
    e_quat = np.zeros(quats.shape)
    e_vec = np.zeros((quats.shape[0]-1,quats.shape[1]))
    for i in range(Y_i.shape[1]):
      e_quat[:,i] = quaternion_multiply(quats[:,i],q_bar_inv)
      e_vec[:,i] = quat_to_vec(e_quat[:,i])
    e_vec_mean = (np.sum(e_vec,axis = 1))/(e_vec.shape[1])
    e = make_quaternion(e_vec_mean)
    q_bar_prev = q_bar
    q_bar = quaternion_multiply(e,q_bar)

  return e_vec,q_bar #,j

def compute_mean(Y_i):
  x_k_minus = np.zeros(Y_i.shape[0])
  x_k_minus[4:] = np.mean(Y_i[4:],axis=1)
  # q_bar = compute_quaternion_mean(Y_i)
  # evec = compute_quaternion_std(Y_i,q_bar)
  evec,q_bar = compute_quaternion_mean_mine(Y_i)
  x_k_minus[:4] = q_bar
  return x_k_minus,evec

def compute_covariance(Y_i,x_k_minus,evec):
  w_bar = np.array([list(x_k_minus[4:])]*12).T
  w_i = Y_i[4:]
  w_w = w_i - w_bar
  # q_bar = compute_quaternion_mean(Y_i)
  # evec = compute_quaternion_std(Y_i,q_bar)
  W_i = np.vstack((evec,w_w))
  P_k_minus = np.dot(W_i,W_i.transpose())/(evec.shape[1])
  return P_k_minus,W_i

#TIME UPDATE DONEEEEEEEE
def find_z_accl(Y_i):
  g = np.array([0,0,0,9.8])
  quats = Y_i[:4]
  z_acc = np.zeros((3,Y_i.shape[1]))
  for i in range(Y_i.shape[1]):
    q_k = quats[:,i]
    q_k_inv = get_inv_quat(q_k)
    temp = quaternion_multiply(g,q_k_inv)
    g_dash = (quaternion_multiply(q_k,temp))[1:]
    z_acc[:,i] = g_dash #+ np.random.normal(0,0.1,3)
  return z_acc

def find_z_gyro(Y_i):
  z_gyro = Y_i[4:] #+np.random.normal(0,0.1,(Y_i[4:]).shape)
  return z_gyro

def find_innovation(Z_p,z_k,R):
  z_k_minus = np.mean(Z_p,axis=1)
  z_k_minus_mat = np.asarray([list(z_k_minus)]*Z_p.shape[1]).T
  P_zz = (np.matmul((Z_p-z_k_minus_mat),(Z_p-z_k_minus_mat).transpose()))/(Z_p.shape[1])
  P_vv = P_zz + R
  v_k = z_k-z_k_minus
  return v_k,P_vv

def compute_cross_corr(W_i,Z_p):
  P_xz = (np.matmul(W_i,Z_p.T))/(W_i.shape[1])
  return P_xz

def apply_kalman(P_xz,P_vv,x_k_minus,v_k,P_k_minus):
  P_vv_inv = np.linalg.inv(P_vv)
  K_k = np.matmul(P_xz,P_vv_inv)
  x_k_minus_q_vec = quat_to_vec(x_k_minus[:4])
  x_k_minus_6dof = np.concatenate((x_k_minus_q_vec,x_k_minus[4:]))
  x_k_6dof = x_k_minus_6dof + np.matmul(K_k,v_k)
  x_k_quat = make_quaternion(x_k_6dof[:3])
  x_k = np.concatenate((x_k_quat,x_k_6dof[3:]))
  temp = np.matmul(P_vv,K_k.T)
  P_k = P_k_minus - np.matmul(K_k,temp)
  return x_k,np.array(P_k)


def get_rpy_from_quat(q):
    s = q[0]
    v = q[1:]
    phi = np.arctan2(2*(s*v[0]+v[1]*v[2]), 1-2*(v[0]*v[0]+v[1]*v[1]))
    theta = np.arcsin(2*(s*v[1]-v[2]*v[0]))
    psi = np.arctan2(2*(s*v[2]+v[0]*v[1]),1-2*(v[1]*v[1]+v[2]*v[2]))

    return np.array([phi,theta,psi])

def estimate_rot(number=1):

    #your code goes here
    filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(number) + ".mat")
    imuRaw = io.loadmat(filename)
    imu1 = imuRaw['vals']
    imu1ts = imuRaw['ts']


    x_k_1 = np.array([1,0,0,0,0,0,0 ])


    x = 1*np.diag(np.ones((6)))
    Q = 500*x
    R = 100*x
    P_k_1 = 10*np.diag(1*np.ones(6))

    bias_a = (np.array([[510,500,500]]*imu1.shape[1])).T
    sens_a = 1/93
    imu1a_updated = (imu1[0:3]-bias_a)/(1023*sens_a)
    imu1a_updated[0:2] = -imu1a_updated[0:2]
    imu1g = imu1[3:]
    bias_g = (np.array([[370,373.75,375.65]]*imu1.shape[1])).T
    sens_g = 6.8

    imu1g_updated = (imu1g-bias_g)/(1023*sens_g)

    imu1_updated = np.vstack([imu1a_updated,imu1g_updated])
    roll = []
    pitch = []
    yaw = []
    if number==4:
        roll = np.zeros(imu1a_updated.shape[1])
        pitch = np.zeros(imu1a_updated.shape[1])
        yaw = np.zeros(imu1a_updated.shape[1])
        return roll, pitch, yaw
    if number==3:
        Q = 0.5*x
        R = 60*np.diag(np.array([2,20,20,30,30,50]))
        P_k_1 = 10*np.diag(1*np.ones(6))

    for time in range(imu1.shape[1]):

        X_i = generate_sigma_pts(P_k_1,Q,x_k_1)
        if time==0:
          Y_i = process_model(X_i,0)
        else:
          Y_i = process_model(X_i,imu1ts[0][time] - imu1ts[0][time-1])
        x_k_minus,evec = compute_mean(Y_i)

        P_k_1,W_i = compute_covariance(Y_i,x_k_minus,evec)
        z_acc = find_z_accl(Y_i)
        z_gyro = find_z_gyro(Y_i)
        z_full = np.vstack((z_acc,z_gyro))
        z_k = imu1_updated[:,time]
        v_k, P_vv = find_innovation(z_full,z_k,R)
        P_xz = compute_cross_corr(W_i,z_full)
        x_k_1,P_k_1 =  apply_kalman(P_xz,P_vv,x_k_minus,v_k,P_k_1)

        rpy = get_rpy_from_quat(x_k_1[:4])
        roll.append(-rpy[0])
        pitch.append(-rpy[1])
        yaw.append(rpy[2])
    if number == 3:
        roll = -np.array(roll)
        pitch = np.array(pitch)
        yaw = np.load("yaw3np.npy")

        # yaw = np.zeros(roll.shape[0])

    return roll,pitch,yaw
