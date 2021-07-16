import os
import unittest

from Models import SimpleProbe, RigidSimpleProbe
from Models import Camera, Imu, n_dofs, dofs_s, dofs_cas

from Filter import Quaternion

import numpy as np
import sympy as sp
t = sp.Symbol('t')

from aux_symbolic import sympy2casadi
from casadi import *

from roboticstoolbox.backends.PyPlot import PyPlot

cam = Camera(filepath='./trajs/offline_mandala0_gt.txt', max_vals=5)

def do_plot(robot, q, elev=None, azim=None):
    env = PyPlot()
    env.launch()
    ax = env.fig.axes[0]
    # elev = None if elev else -90
    # azim = azim if azim else -90
    ax.view_init(elev=elev, azim=azim)
    env.add(robot)
    robot.q = q
    env.hold()

def view_selector(robot, q):
    import time
    env = PyPlot()
    env.launch()
    env.add(robot)

    ax = env.fig.axes[0]
    # xy (90, 270)
    elev, azim = 30, -60

    do_switch = False
    params = ['elev', 'azim']
    param = params[0]

    try:
        while True:
            print(f"Current elev: {elev}, current azim: {azim}.")
            ax.view_init(elev=elev, azim=azim)

            robot.q = q
            env.step()

            ans = input(f"Next {param}?")
            try:
                elev = int(ans) if (param == 'elev') else elev
                azim = int(ans) if (param == 'azim') else azim
            except ValueError:
                if ans:
                    do_switch = True

            if do_switch:
                param = [p for p in params if param != p][0]
                do_switch = False
    except:
        sys.exit()

class TestSymbolic(unittest.TestCase):
    """ Module to troubleshoot symbolic stuff. """

    @classmethod
    def setUpClass(cls):
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        # cls.acc_CB = -cls.probe.acc
        cls.R_s = sp.MatrixSymbol('R_BW', 3, 3)
        cls.acc_C_s = sp.MatrixSymbol('acc_C', 3, 1)

        cls.q_s = [sp.Symbol(f'q{x}') for x in range(1,n_dofs+1)]
        cls.q_dot_s = [sp.Symbol(f'q{x}_dot') for x in range(1,n_dofs+1)]
        cls.q_ddot_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,n_dofs+1)]

        cls.params = [*cls.q_s, *cls.q_dot_s, *cls.q_ddot_s]

    def gen_partial_expr(self):
        q1dd = sp.symbols('q1_ddot')
        q5dd = sp.symbols('q5_ddot')
        q9 = sp.symbols('q9')

        return -q1dd*sp.sqrt(3)*sp.sin(q9) \
                + 1.0*q5dd

    def custom_sin(self, arg):
        return sp.sin(arg)

    def test_partial_expr(self):
        partial_expr = self.gen_partial_expr()
        func = sp.lambdify(self.params, partial_expr,
                # modules=[{'sin': custom_sin}, 'math'],
                # modules='math',
                modules='sympy',
                )

        res = func(*self.q_s, *self.q_dot_s, *self.q_ddot_s)
        print(res)

    def test_tensor_mult(self):
        from sympy.abc import a, b, c, d, e, f, g, h, i, j, k, l, m, n, o
        from sympy.tensor.array import tensorproduct
        from sympy.tensor.array import tensorcontraction

        H = sp.MutableDenseNDimArray([0]*(2*3*4), (2,3,4))
        H[0,:,:] = sp.Matrix([[a, b, c, d], [e, f, g, h], [m, n, c, d]])
        H[1,:,:] = sp.Matrix([[e, f, g, h], [a, b, c, d], [e, f, g, h]])

        qd1 = sp.MutableDenseNDimArray([i, j, k, l], (4,1))

        tp1 = tensorproduct(H, qd1)
        assert(tp1.shape == (2, 3, 4, 4, 1))
        tp1 = tensorcontraction(tp1, (2,3))
        assert(tp1.shape == (2, 3, 1))
        tp1 = tp1[:,:,0]
        assert(tp1.shape == (2, 3))

        qd2 = sp.MutableDenseNDimArray([m, n, o], (3,1))
        tp2 = tensorproduct(tp1, qd2)
        assert(tp2.shape == (2, 3, 3, 1))
        tp2 = tensorcontraction(tp2, (1,2))
        assert(tp2.shape == (2, 1))

    def test_expr_with_acc(self):
        expr = self.probe.acc
        expr = self.probe.alp

class TestCasadi(unittest.TestCase):
    def setUp(self):
        """ Initialise probe and joint variables. """
        self.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)

    def test_sympy2casadi(self):
        x,y = sp.symbols("x y")
        xy = sp.Matrix([x,y])
        e = sp.Matrix([x*sp.sqrt(y),sp.sin(x+y),abs(x-y)])

        X = SX.sym("xc")
        Y = SX.sym("yc")
        XY = casadi.vertcat(X,Y)

        res = sympy2casadi(e, [x, y], XY)
        print(res)

    def test_probe2casadi(self):
        T = self.probe.R
        T = self.probe.p
        J = self.probe.jacob0(self.probe.q_s)
        T = J @ dofs_s[n_dofs:2*n_dofs]
        T = self.probe._calc_acceleration()
        T = sympy2casadi(T, dofs_s, dofs_cas)

class TestSimpleProbeBC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # probe and joint variables.
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0. for q in cls.probe.q_s]

    @unittest.skip("Skip plot.")
    def test_plot(self):
        view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)

    @unittest.skip("Not necessary after implementing CasADi.")
    def test_lambdify_R(self):
        """ Ensures correct substitutions of D.O.F.s in the expression
            for R. """
        q7 = sp.Symbol('q7')
        R_func = sp.lambdify(q7, self.probe.R, 'numpy')

class TestRigidSimpleProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initialise probe and joint variables. """
        cls.probe = RigidSimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0. for q in cls.probe.q_s]
        cls.qd = [sp.diff(q, t) for q in cls.q_0]
        cls.qdd = [sp.diff(q, t) for q in cls.qd]
        cls.joint_dofs = [*cls.q_0, *cls.qd, *cls.qdd]

        # parameters from fwkin
        cls.R_BC = cls.probe.R

    @unittest.skip("Skip plot.")
    def test_plot(self):
        # view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)

class TestImu(TestRigidSimpleProbe):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.filepath = './trajs/imu_test.txt'

        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)

    def setUp(self):
        self.imu = Imu(self.probe, cam)

    def _get_numlines(self, filepath):
        try:
            with open(filepath) as f:
                n = sum(1 for line in f)
            return n
        except FileNotFoundError:
            return 0

    def test_append_array(self):
        assert(self.imu._om == [])

        for i in range(2):
            self.imu.eval_expr_single(cam.t[i], self.probe.q_cas,
                self.probe.qd_cas, self.probe.qdd_cas,
                cam.acc[:,i], cam.R[i],
                cam.om[:,i], cam.alp[:,i], append_array=True)

        assert(self.imu.om.shape == (3, 2))

    def test_write_array_to_file(self):
        self.test_append_array()
        self.imu.write_array_to_file(self.filepath)

        num_lines = self._get_numlines(self.filepath)
        self.assertEqual(num_lines, 2)

    def test_append_file(self):
        num_lines = self._get_numlines(self.filepath)

        num_data = 3
        for i in range(num_data):
            self.imu.eval_expr_single(cam.t[i], self.probe.q_cas,
                self.probe.qd_cas, self.probe.qdd_cas,
                cam.acc[:,i], cam.R[i], cam.om[:,i],
                cam.alp[:,i], append_array=False, filepath=self.filepath)

        new_num_lines = self._get_numlines(self.filepath)
        self.assertEqual(new_num_lines, num_lines + num_data)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)
        super().tearDownClass()

class TestQuaternions(unittest.TestCase):
    def setUp(self):
        self.q1 = Quaternion(x=0, y=-0.002, z=-0.001, w=1)
        self.q2 = Quaternion(x=0, y=0, z=0, w=1)

class TestFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dt = casadi.SX.sym('dt')

        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        print(cls.probe)
        print(f'q: {cls.probe.q}\n')

        # camera and imu sensors
        # num_imu_between_frames = 1
        # cam_interp = cam.interpolate(num_imu_between_frames)
        min_t, max_t = cam.t[0], cam.t[-1]

        # imu
        imu = Imu(cls.probe, cam)
        imu.eval_init(cls.probe.q, cls.probe.qd, cls.probe.qdd)
        cls.imu = imu

        # fwkin
        cls.p_CB, cls.v_CB, cls.acc_CB = cls.probe.p, cls.probe.v, cls.probe.acc
        cls.R_BC, cls.om_CB, cls.alp_CB = cls.probe.R, cls.probe.om, cls.probe.alp

        cls.states()
        cls.error_states()
        cls.inputs()
        cls.measurements()
        cls.noise()

        # long equations
        cls.err_p_C_dot = cls._derive_err_pc_dot()
        cls.f_err_p_C_dot = casadi.Function('f_err_p_C_dot',
                [cls.dt, *cls.err_x, *cls.u, *cls.n, cls.R_WB],
                [cls.err_p_C_dot],
                ['dt', *cls.err_x_str, *cls.u_str,
                    *cls.n_str, 'R_WB'],
                ['err_p_C_dot'])

    @classmethod
    def states(cls):
        cls.p_B = casadi.SX.sym('p_B', 3)
        cls.v_B = casadi.SX.sym('v_B', 3)
        cls.R_WB = casadi.SX.sym('R_WB', 3, 3)

         # note: creating cls.dofs via casadi.SX.sym results in free variables in the functions created later on -- therefore dofs_cas, which was used in Probe.py, has to be imported
        cls.dofs, cls.ddofs, cls.dddofs = casadi.vertsplit(dofs_cas, [0, 8, 16, 24])
        cls.dofs_t, cls.dofs_r, _ = casadi.vertsplit(cls.dofs, [0, 3, 6, 8])

        cls.p_C = casadi.SX.sym('p_C', 3)
        cls.R_WC = casadi.SX.sym('p_C', 3, 3)

        cls.x = [cls.p_B, cls.v_B, cls.R_WB, cls.dofs_t, cls.dofs_r, cls.p_C, cls.R_WC]
        cls.x_str = ['p_B', 'v_B', 'R_WB', 'dofs_t', 'dofs_r', 'p_C', 'R_WC']

    @classmethod
    def error_states(cls):
        cls.err_p_B = casadi.SX.sym('err_p_B', 3)
        cls.err_v_B = casadi.SX.sym('err_v_B', 3)
        cls.err_theta = casadi.SX.sym('err_theta', 3)
        cls.err_dofs_t = casadi.SX.sym('err_dofs_t', 3)
        cls.err_dofs_r = casadi.SX.sym('err_dofs_r', 3)
        cls.err_p_C = casadi.SX.sym('err_p_C', 3)
        cls.err_theta_C = casadi.SX.sym('err_theta_C', 3)

        cls.err_x = [cls.err_p_B, cls.err_v_B, cls.err_theta,
                    cls.err_dofs_t, cls.err_dofs_r, cls.err_p_C, cls.err_theta_C]
        cls.err_x_cas = casadi.vertcat(*cls.err_x)
        cls.err_x_str = ['err_p_B', 'err_v_B', 'err_theta',
                    'err_dofs_t', 'err_dofs_r', 'err_p_C', 'err_theta_C']

    @classmethod
    def measurements(cls):
        _, cls.q_notch, _ = casadi.vertsplit(cls.dofs, [0, 6, 7, 8])
        _, cls.qd_notch, _ = casadi.vertsplit(cls.ddofs, [0, 6, 7, 8])
        cls.notch_dof = [cls.q_notch, cls.qd_notch]
        cls.notch_dof_str = ['q_notch', 'qd_notch']

    @classmethod
    def inputs(cls):
        cls.acc = casadi.SX.sym('acc', 3)
        cls.om = casadi.SX.sym('om', 3)

        cls.u = [cls.om, cls.acc]
        cls.u_str = ['om', 'acc']

    @classmethod
    def noise(cls):
        cls.n_v = casadi.SX.sym('n_v', 3)
        cls.n_om = casadi.SX.sym('n_om', 3)
        cls.n_dofs_t = casadi.SX.sym('n_dofs_t', 3)
        cls.n_dofs_r = casadi.SX.sym('n_dofs_r', 3)

        cls.n = [cls.n_v, cls.n_om, cls.n_dofs_t, cls.n_dofs_r]
        cls.n_cas = casadi.vertcat(*cls.n)
        cls.n_str = ['n_v', 'n_om', 'n_dofs_t', 'n_dofs_r']

    @classmethod
    def _derive_err_pc_dot(cls):
        """
            Example derivation of err_p_B:

            [In continuous time]
            p_B_tr_dot = p_B_dot + err_p_B_dot
            v_B_tr = v_B + err_v_B

            err_p_B_dot = v_B_tr - v_B
                        = v_B + err_v_B - v_B
                        = err_v_B

            [Discretised]
            err_p_B_next = err_p_B + dt * err_v_B
        """

        # deriving err_p_C_dot -- define the true values
        v_B_tr = cls.v_B + cls.err_v_B
        R_WB_tr = cls.R_WB @ (casadi.DM.eye(3) \
                    + casadi.skew(cls.err_theta))
        dofs_t_tr = cls.dofs_t + cls.err_dofs_t
        dofs_r_tr = cls.dofs_r + cls.err_dofs_r # this is a placeholder ## TODO
        dofs_tr = casadi.vertcat(dofs_t_tr, dofs_r_tr)
        om_tr = cls.om - cls.n_om

        # deriving err_p_C_dot -- continuous time
        p_CB_dot = cls.R_WB @ cls.v_CB \
                + casadi.skew(cls.om) @ cls.R_WB @ cls.p_CB
        p_CB_dot_tr = R_WB_tr @ cls.v_CB \
                + casadi.skew(om_tr) @ R_WB_tr @ cls.p_CB

        p_C_dot = cls.v_B + p_CB_dot
        p_C_dot_tr = v_B_tr + p_CB_dot_tr

        # err_p_C_dot = p_C_dot_tr - p_C_dot # results in free variables v_B
        err_p_C_dot = cls.err_v_B + p_CB_dot_tr - p_CB_dot

        return err_p_C_dot

    def test_fun_nominal(self):
        p_B_next = self.p_B \
                + self.dt * self.v_B \
                + self.dt**2 / 2 * self.R_WB @ self.acc
        om_C = self.R_BC.T @ self.om + self.R_BC.T @ self.om_CB

        fun_nominal = casadi.Function('f_nom',
            [self.dt, *self.x, *self.u, *self.notch_dof],
            [   p_B_next,
                self.v_B + self.dt * self.R_WB @ self.acc,
                self.R_WB + self.R_WB @ casadi.skew(self.dt * self.om),
                self.dofs_t,
                self.dofs_r,
                self.p_C + self.dt * self.v_B \
                    + self.dt**2 / 2 * self.R_WB @ self.acc \
                    + self.R_WB @ self.p_CB,
                self.R_WC + self.R_WC @ casadi.skew(self.dt * om_C)],
            ['dt', *self.x_str, *self.u_str, *self.notch_dof_str],
            ['p_B_next', 'v_B_next', 'R_WB_next',
                'dofs_t_next', 'dofs_r_next', 'p_C_next', 'R_WC_next'])

        res = fun_nominal(  dt  = 0.1,
                            p_B = casadi.DM([1.2, 3.9, 2.]),
                            v_B = casadi.DM([0.01, 0.02, 0.003]),
                            R_WB = casadi.DM.eye(3),
                            om = casadi.DM(self.imu.om),
                            acc = casadi.DM(self.imu.acc),
                            q_notch = casadi.DM(0.),
                            qd_notch = casadi.DM(0.),
                         )
        p_B_next = res['p_B_next']

    def _gen_trmatr_err_p_C_dot(self):
        jac_err_p_C_dot = self.f_err_p_C_dot.jac()
        res = jac_err_p_C_dot(  dt   = self.dt,
                                om   = self.om,
                                acc  = self.acc,
                                R_WB = self.R_WB)

        # jacobian of err_p_cam w.r.t. err_x
        l_in = 0
        r_in = 0

        trmatr_err_p_C_dot = np.zeros((self.err_p_C_dot.shape[0],
                            self.err_x_cas.shape[0]))

        for x in self.err_x_str:
            name = 'Derr_p_C_dotD' + x
            res_np = casadi.DM(res[name]).full()

            r_in += res_np.shape[1]
            trmatr_err_p_C_dot[0:3,l_in:r_in] = res_np
            l_in = r_in

        return trmatr_err_p_C_dot

    def _gen_nmatr_err_p_C_dot(self):
        jac_err_p_C_dot = self.f_err_p_C_dot.jac()
        res = jac_err_p_C_dot(  dt   = self.dt,
                                om   = self.om,
                                acc  = self.acc,
                                R_WB = self.R_WB)

        # jacobian of err_p_cam w.r.t. n
        l_in = 0
        r_in = 0

        nmatr_err_p_C_dot = np.zeros((self.err_p_C_dot.shape[0],
                self.n_cas.shape[0]))

        for n in self.n_str:
            name = 'Derr_p_C_dotD' + n
            res_np = casadi.DM(res[name]).full()

            r_in += res_np.shape[1]
            nmatr_err_p_C_dot[0:3,l_in:r_in] = res_np
            l_in = r_in

        return nmatr_err_p_C_dot

    def test_fun_error(self):
        trmatr_err_p_C_dot = self._gen_trmatr_err_p_C_dot()

        err_p_C_next = self.err_p_C \
                + self.dt @ trmatr_err_p_C_dot @ self.err_x_cas

        fun_error = casadi.Function('f_err',
            [self.dt, *self.err_x, *self.u, *self.n, self.R_WB],
            [   self.err_p_B + self.dt * self.err_v_B,
                self.err_v_B + self.dt * (-self.R_WB @ casadi.skew(self.acc) @ self.err_theta) + self.n_v,
                -casadi.cross(self.om, self.err_theta) + self.n_om,
                self.n_dofs_t,
                self.n_dofs_r,
                err_p_C_next ],
            ['dt', *self.err_x_str, *self.u_str,
                *self.n_str, 'R_WB'],
            ['err_p_B_next', 'err_v_B_next', 'err_theta_next',
                'err_dofs_t_next', 'err_dofs_r_next',
                'err_p_C_next'])

        res = fun_error(  dt  = 0.1,
                        err_p_B = casadi.DM([1.2, 3.9, 2.]),
                        err_v_B = casadi.DM([0.01, 0.02, 0.003]),
                        R_WB = casadi.DM.eye(3),
                        om = casadi.DM(self.imu.om),
                        acc = casadi.DM(self.imu.acc),
                         )
        err_p_B_next = res['err_p_B_next']

    def test_covariance(self):
        trmatr_err_p_C_dot = self._gen_trmatr_err_p_C_dot()
        nmatr_err_p_C_dot = self._gen_nmatr_err_p_C_dot()

        Fx = casadi.SX.eye(self.err_x_cas.shape[0])

        Fx[0:3, 3:6] = self.dt * np.eye(3)
        Fx[3:6, 6:9] = - self.R_WB @ casadi.skew(self.acc) * self.dt
        Fx[6:9, 6:9] = casadi.skew(self.om * self.dt).T
        Fx[9:15, :]  = 0 # dofs
        Fx[15:18, :] = Fx[15:18, :] + self.dt * trmatr_err_p_C_dot

        Fi = casadi.SX.zeros(self.err_x_cas.shape[0], self.n_cas.shape[0])
        Fi[3:15, :] = casadi.SX.eye(12)
        Fi[15:18, :] = nmatr_err_p_C_dot

        P0 = casadi.SX.eye(self.err_x_cas.shape[0]) * 0.01
        Q = casadi.SX.eye(self.n_cas.shape[0]) * 0.01

        P = Fx @ P0 @ Fx.T  + Fi @ Q @ Fi.T

def suite():
    suite = unittest.TestSuite()
    test_class = TestSimpleProbeBC
    for t in test_class.__dict__.keys():
        if t.startswith('test'):
            suite.addTest(test_class(t))
    return suite

if __name__ == '__main__':
    # run all tests
    unittest.main(verbosity=2)

    # run only certain tests
    runner = unittest.TextTestRunner()
    runner.run(suite())