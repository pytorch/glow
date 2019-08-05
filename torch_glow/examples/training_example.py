import torch
import torch_glow
import torch.nn.functional as F
import utils.torchvision_fake.resnet as R
import torch.optim as optim

def foo(x, y):
    return 2 * x + y


traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
print("simple tensor jit ir")
print(traced_foo.graph_for(torch.rand(3, requires_grad=False), torch.rand(3, requires_grad=False)))
print("simple tensor with grad jit ir")
print(traced_foo.graph_for(torch.rand(3, requires_grad=True), torch.rand(3, requires_grad=True)))


model = R.resnet18()
data = torch.randn(1, 3, 224, 224, requires_grad=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
out = model(data)
criterion = torch.nn.MSELoss()
target = torch.randn(1, 1000, dtype=torch.float)
loss = criterion(out, target)
loss.backward()
optimizer.step()
traced_resnet18 = torch.jit.trace(model, data)
print("resnet18 with 1 step of backprop jit ir")
print(traced_resnet18.graph_for(data))
