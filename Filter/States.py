import numpy as np
from .Quaternion import Quaternion

class States(object):
    def __init__(self, p, v, q, dofs, p_cam, q_cam):
        self._p = np.asarray(p).reshape(3,)
        self._v = np.asarray(v).reshape(3,)
        self._q = Quaternion(val=q, do_normalise=True)
        self._dofs = dofs
        self._p_cam = p_cam.reshape(3,)
        self._q_cam = Quaternion(val=q_cam, do_normalise=True)

        self.size = len(p) + len(v) + len(self.q.xyzw) \
                    + len(dofs) + len(p_cam) + len(self.q_cam.xyzw)
        assert(self.size == 23)

    def apply_correction(self, err):
        self.p += err.dp.reshape(3,)
        self.v += err.dv.reshape(3,)
        self.q = self.q * err.dq
        self.q.normalise()
        self.dofs += err.ddofs.reshape(6,)
        self.p_cam += err.dpc.reshape(3,)
        self.q_cam = self.q_cam * err.dqc
        self.q_cam.normalise()

    def set(self, vec):
        self.p = vec[0].squeeze()
        self.v = vec[1].squeeze()
        self.q = vec[2].squeeze()
        self.dofs = vec[3].squeeze()
        self.p_cam = vec[4].squeeze()
        self.q_cam = vec[5].squeeze()

    def __repr__(self):
        return f'State: p_cam ({self._p_cam}), ...'

    @property
    def vec(self):
        return [self.p, self.v, self.q.rot,
                    self.dofs,
                    self.p_cam, self.q_cam.rot]

    @property
    def p(self):
        return self._p.copy()

    @p.setter
    def p(self, val):
        self._p = val

    @property
    def v(self):
        return self._v.copy()

    @v.setter
    def v(self, val):
        self._v = val

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = Quaternion(val=val, do_normalise=True)

    @property
    def dofs(self):
        return self._dofs.copy()

    @dofs.setter
    def dofs(self, val):
        self._dofs = val

    @property
    def p_cam(self):
        return self._p_cam.copy()

    @p_cam.setter
    def p_cam(self, val):
        self._p_cam = val

    @property
    def q_cam(self):
        return self._q_cam

    @q_cam.setter
    def q_cam(self, val):
        self._q_cam = Quaternion(val=val, do_normalise=True)

class ErrorStates(object):
    def __init__(self, vec):
        self.set(vec)

    def set(self, vec):
        self.vec = vec
        self.size = len(vec)
        assert(self.size == 21)

        p = vec[0:3]
        v = vec[3:6]
        theta = vec[6:9]
        dofs = vec[9:15]
        p_c = vec[15:18]
        theta_c = vec[18:]

        self.dp = np.asarray(p)
        self.dv = np.asarray(v)
        self.dq = Quaternion(v=np.array(theta)/2, w=1.)
        self.ddofs = np.asarray(dofs)
        self.dpc = np.asarray(p_c)
        self.dqc = Quaternion(v=np.array(theta_c)/2, w=1.)

        self.theta = np.asarray(theta)
        self.theta_c = np.asarray(theta_c)

    def reset(self):
        self.set([0] * 21)