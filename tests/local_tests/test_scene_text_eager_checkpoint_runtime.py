"""Check the launch-time workaround without weakening checkpoint checks."""
import os
import subprocess
import sys


def test_eager_runtime_preserves_variable_length_checkpoint_gradients():
    script = '''
import torch
from torch.utils.checkpoint import checkpoint
torch.manual_seed(23)
compiler_calls = []
def backend(graph, inputs):
    compiler_calls.append(True)
    return graph.forward
weight = torch.randn(16, 16)
a = torch.randn(4, 16, requires_grad=True)
b = torch.randn(16, 4, requires_grad=True)
def forward(x):
    assert not torch.compiler.is_compiling()
    return torch.nn.functional.silu(x @ weight.T + x @ a.T @ b.T)
compiled = torch.compile(forward, backend=backend)
with torch.no_grad():
    for length in range(2, 14):
        compiled(torch.randn(length, 16))
for length in [3, 7, 5, 11, 3]:
    x = torch.randn(length, 16, requires_grad=True)
    reference = torch.autograd.grad(forward(x).square().mean(), (x, a, b))
    result = checkpoint(compiled, x, use_reentrant=False)
    actual = torch.autograd.grad(result.square().mean(), (x, a, b))
    for left, right in zip(reference, actual):
        torch.testing.assert_close(left, right)
assert not compiler_calls, compiler_calls
print('EAGER_CHECKPOINT_VARIABLE_TEXT_GRADIENTS_PASS')
'''
    result = subprocess.run(
        [sys.executable, '-c', script], check=True, text=True, capture_output=True, timeout=120,
        env=dict(os.environ, TORCHDYNAMO_DISABLE='1', TORCH_COMPILE_DISABLE='1',
                 CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    assert 'EAGER_CHECKPOINT_VARIABLE_TEXT_GRADIENTS_PASS' in result.stdout
