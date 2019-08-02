import torch

from tests.utils import jitVsGlow


def test_addmm():

    def test_basic(M, aa, b, bias):
        a = aa.t()
        M1 = M.addmm(a, b)
        M2 = M.add(bias)
        return M2

    x = torch.randn(10, 6)
    y = torch.randn(10, 6)
    z = torch.randn(6, 6)
    a = torch.randn(6, 6)

    jitVsGlow(test_basic, z, x, y, a)
