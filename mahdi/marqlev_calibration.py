import numpy as np
from scipy.optimize import fsolve
import cv2
from scipy import optimize

p_c_0 = np.load("/home/user/code/zed-sdk/mahdi/log/marqlev_calib/p_c_0.npy")
p_c_1 = np.load("/home/user/code/zed-sdk/mahdi/log/marqlev_calib/p_c_1.npy")
p_c_2 = np.load("/home/user/code/zed-sdk/mahdi/log/marqlev_calib/p_c_2.npy")
p_c_3 = np.load("/home/user/code/zed-sdk/mahdi/log/marqlev_calib/p_c_3.npy")

p_o_3 = np.array([215, 88, 25]) / 1000
p_o_2 = p_o_3 + np.array([-36.1, 0, 0]) / 1000
p_o_1 = p_o_3 + np.array([-36.1, 36.1, 0]) / 1000
p_o_0 = p_o_3 + np.array([0, 36.1, 0]) / 1000

O_T_EE = np.load("/home/user/code/zed-sdk/mahdi/log/marqlev_calib/O_T_EE.npy")
inv_O_T_EE = np.linalg.inv(O_T_EE)

# chessboard solution
R_cam2gripper = np.array([[-0.00715975, 0.68714525, -0.72648478],
                          [-0.99971785, -0.0213735, -0.01036357],
                          [-0.0226488, 0.7262056, 0.6871044]])
t_cam2gripper = np.array([[0.1266802],
                          [0.02387603],
                          [-0.11579095]])
r0 = cv2.Rodrigues(R_cam2gripper)[0]
t0 = t_cam2gripper


def func(x):
    theta = np.sqrt(x[3] ** 2 + x[4] ** 2 + x[5] ** 2)
    u1 = x[3] / theta
    u2 = x[4] / theta
    u3 = x[5] / theta

    C = np.cos(theta)
    S = np.sin(theta)

    H = np.array([[C + (1 - C) * u1 ** 2, (1 - C) * u1 * u2 - u3 * S, (1 - C) * u1 * u3 + u2 * S, x[0]],
                  [(1 - C) * u2 * u1 + u3 * S, C + (1 - C) * u2 ** 2, (1 - C) * u2 * u3 - u1 * S, x[1]],
                  [(1 - C) * u3 * u1 - u2 * S, (1 - C) * u3 * u2 + u1 * S, C + (1 - C) * u3 ** 2, x[2]],
                  [0, 0, 0, 1]])

    out3 = np.matmul(H, np.append(p_c_3, 1)) - np.matmul(inv_O_T_EE, np.append(p_o_3, 1))
    out2 = np.matmul(H, np.append(p_c_0, 1)) - np.matmul(inv_O_T_EE, np.append(p_o_0, 1))

    return np.append(out3[:3], out2[:3])


x0 = np.append(t0, r0)
# root, infodict, ier, mesg = fsolve(func, x0, full_output=True)
# print("root=", root)
# print("ier=", ier)
# print("mesg=", mesg)
# print("infodict=", infodict)
# print(np.isclose(func(root), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

sol = optimize.root(func, x0, jac=False, method='lm')
print(sol.x)
print(np.isclose(func(sol.x), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
