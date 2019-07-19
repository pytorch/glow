import torch

from tests.utils import jitVsGlow

 # Test of PyTorch t (transpose) on Glow with 2d inputs.
def test_transpose_2d():

   def test_f(a):
        b = a + a
        return b.t()

   x = torch.randn(7, 4)

   jitVsGlow(test_f, x)

# Test of PyTorch t (transpose) on Glow with 1d inputs.
def test_transpose_1d():

   def test_f(a):
        b = a + a
        return b.t()

   x = torch.randn(7)

   jitVsGlow(test_f, x)


# Test of PyTorch t_ (in place transpose) on Glow.
def test_transpose_inplace():

   def test_f(a):
        b = a + a
        return b.t_()

   x = torch.randn(7, 4)

   jitVsGlow(test_f, x)
