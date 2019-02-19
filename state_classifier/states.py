from strawberryfields.ops import Preparation, Catstate
from strawberryfields.parameters import (abs, sqrt, transpose, squeeze, _unwrap)

class ONstate(Preparation):
    def __init__(self, n, delta=1.0):
        super().__init__([n, delta])

    def _apply(self, reg, backend, **kwargs):
        n = int(self.p[0].x)
        delta = self.p[1]

        D = backend.get_cutoff_dim()
        vac = np.zeros(D)[:, np.newaxis]
        vac[0] = 1.0

        fock = np.zeros(D)[:, np.newaxis]
        fock[n] = 1

        N = 1.0 / sqrt(1 + abs(delta)**2)

        ket = (vac + delta*fock) * N
        ket = transpose(ket)
        ket = squeeze(ket)

        backend.prepare_ket_state(ket.x, *reg)

class SqueezedCatstate(Catstate):
    def __init__(self, alpha=0, p=0, r=0):
        super(Catstate, self).__init__([alpha, p, r])

    def _apply(self, reg, backend, **kwargs):
        super()._apply(reg, backend, **kwargs)
        p = _unwrap(self.p)
        backend.squeeze(p[2], *reg)
