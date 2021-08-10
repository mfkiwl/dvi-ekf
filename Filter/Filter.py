from copy import copy
from math import factorial
import numpy as np

from .Quaternion import Quaternion, skew
from .Trajectory import FilterTraj
from .States import States, ErrorStates

import casadi
from . import context
import symbols as sym
import symbolic_eqns as eqns

class Filter(object):
    def __init__(self, imu, probe, IC, P0, meas_noise):
        self.num_states = IC.size
        self.num_error_states = IC.size - 2
        self.num_meas = 7
        self.num_noise = 12

        self.states = copy(IC)
        # self.err_states = ErrorStates([0] * self.num_error_states) # defaults to zero
        self._x = []
        self._u = []

        self.dt = 0.
        self.traj = FilterTraj("kf")

        # imu / noise
        self.imu = imu
        self.probe = probe

        self.stdev_na = np.array(imu.stdev_na)
        self.stdev_nom = np.array(imu.stdev_nom)
        self.R = np.diag(meas_noise)

        # buffer
        self._om_old = imu.om.squeeze()
        self._acc_old = imu.acc.squeeze()
        self.R_WB_old = self.states.q.rot

        # covariance
        self.P = P0
        assert(self.P.shape == (self.num_error_states, self.num_error_states))

        # static matrices
        self.Hx = np.zeros([self.num_meas, self.num_states])
        self.Hx[:3,-6:-3] = np.eye(3)
        self.Hx[3:7,-4:] = np.eye(4)

    @property
    def x(self):
        self._x = self.states.vec
        return self._x

    @property
    def u(self):
        self._u = [self.om_old, self.acc_old]
        return self._u

    @property
    def om_old(self):
        return self._om_old

    @om_old.setter
    def om_old(self, val):
        self._om_old = val.squeeze()

    @property
    def Om_old(self):
        return Quaternion(w=1., v=(0.5 * self.dt * self.om_old) )

    @property
    def acc_old(self):
        return self._acc_old

    @acc_old.setter
    def acc_old(self, val):
        self._acc_old = val.squeeze()

    @property
    def jac_X_deltx(self):
        X_deltx = np.zeros((self.num_states, self.num_error_states))

        def jac_true_wrt_err_quats(quat):
            x, y, z, w = self.states.q.xyzw
            return 0.5 * np.array([[-x, -y, -z],
                                   [ w, -z,  y],
                                   [ z,  w, -x],
                                   [-y,  x,  w]])

        Q_deltth = jac_true_wrt_err_quats(self.states.q)
        Q_deltth_C = jac_true_wrt_err_quats(self.states.q_cam)

        X_deltx[0:6,0:6] = np.eye(6)
        X_deltx[6:10,6:9] = Q_deltth
        X_deltx[10:19,9:18] = np.eye(9)
        X_deltx[19:23,18:21] = Q_deltth_C

        return X_deltx

    def propagate(self, t, om, acc, do_prop_only=False):
        self._predict_nominal()
        self._predict_error()
        self._predict_error_covariance()

        # Buffer
        self.om_old = om
        self.acc_old = acc
        self.R_WB_old = self.states.q.rot

        # for plotting
        self.traj.append_state(t, self.states)

    def _predict_nominal(self):
        res = [casadi.DM(r).full() \
                    for r in eqns.f_predict(self.dt,
                        *self.x,
                        *self.u,
                        *self.probe.fwkin)]
        self.states.set(res)

    def _predict_error(self):
        """ Calculates jacobian of the error state kinematics w.r.t. error states and w.r.t. noise. """

        Fx = casadi.SX.eye(self.num_error_states)
        Fx[0:3, 3:6] = self.dt * casadi.SX.eye(3)
        Fx[3:6, 6:9] = - self.R_WB_old @ casadi.skew(self.acc_old) * self.dt
        Fx[6:9, 6:9] = self.Om_old.rot.T
        # Fx[9:15, :] # dofs
        self.Fx = self._cam_error_jacobian(Fx, sym.err_x)

        # motion model noise jacobian
        Fi = casadi.SX.zeros(self.num_error_states, self.num_noise)
        Fi[3:15, :] = casadi.SX.eye(self.num_noise)
        self.Fi = self._cam_error_jacobian(Fi, sym.n)

        """ returns zero
        self.err_states.set(self.Fx @ self.err_states.vec) """

    def _cam_error_jacobian(self, jac, vars_wrt):
        """ Fills the error jacobian (either w.r.t. error state or
            w.r.t. noise) for the camera state entries. """

        err_p_C_next = sym.err_p_C \
                + sym.dt * sym.get_err_pc_dot(self.probe)
        err_theta_C_next = sym.err_theta_C \
                + sym.dt * sym.get_err_theta_c_dot(self.probe)

        l_in = 0
        r_in = 0
        for x in vars_wrt:
            r_in += x.shape[0]
            jac[15:18,l_in:r_in] = casadi.jacobian(err_p_C_next, x)
            jac[18:,l_in:r_in] = casadi.jacobian(err_theta_C_next, x)
            l_in = r_in

        fun_jac = casadi.Function('f_jac',
            [sym.dt, sym.dofs, sym.err_dofs, sym.R_WB, *sym.u, sym.n_om, sym.err_theta, sym.err_theta_C], [jac],
            ['dt', 'dofs', 'err_dofs', 'R_WB',
                *sym.u_str, 'n_om', 'err_theta', 'err_theta_C'], ['jac']
            )
        return casadi.DM(
                fun_jac( dt         = self.dt,
                        dofs        = self.states.dofs,
                        err_dofs    = casadi.DM.zeros(6,),
                        R_WB        = self.R_WB_old,
                        B_om_BW     = self.om_old,
                        B_acc_BW    = self.acc_old,
                        n_om        = self.imu.stdev_nom,
                        err_theta   = casadi.DM([0., 0., 0.]),
                        err_theta_C = casadi.DM([0., 0., 0.]),
                        )['jac']).full()

    def _predict_error_covariance(self):
        Q = np.eye(self.num_noise)
        Q[0:3, 0:3] = self.dt**2 * self.stdev_na**2 * np.eye(3)
        Q[3:6, 3:6] = self.dt**2 * self.stdev_nom**2 * np.eye(3)

        sigma_dofs_p = 0.02
        sigma_dofs_r = 0.1
        N_p = np.random.normal(loc=0, scale=sigma_dofs_p, size=(3,))
        N_r = np.random.normal(loc=0, scale=sigma_dofs_r, size=(3,))
        Q[6:12, 6:12] = np.diag(np.hstack((N_r, N_p)))

        self.P = self.Fx @ self.P @ self.Fx.T + self.Fi @ Q @ self.Fi.T

    def update(self, camera):
        # compute gain        
        H = self.Hx @ self.jac_X_deltx # 7x21
        S = H @ self.P @ H.T + self.R # 7x7
        try:
            K = self.P @ H.T @ np.linalg.inv(S) # 21x7
        except np.linalg.LinAlgError as e:
            print(f"ERROR: {e}!")
            print("Stopping simulation.")
            return None

        # compute error state
        res_p_cam = camera.pos.reshape((3,)) - self.states.p_cam.reshape((3,))
        res_q = (camera.qrot - self.states.q_cam).xyzw
        res = np.hstack((res_p_cam, res_q))
        err = ErrorStates(K @ res)

        # correct predicted state and covariance
        self.states.apply_correction(err)
        I = np.eye(self.num_error_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        # reset error states
        G = np.eye(self.num_error_states)
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * err.theta)
        G[-3:, -3:] = np.eye(3) - skew(0.5 * err.theta_c)
        self.P = G @ self.P @ G.T

        return K