import os
import sys
import math

# ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import torch.nn.functional as F
from models.vgg import vgg16_vfl
from models.global_models import ClassificationModelHostTrainableHead


EXPECTED_EMBED_DIM = 10752
NUM_CLASSES = 10


def run_check(bs=2, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)

    m0 = vgg16_vfl(NUM_CLASSES).to(device).train()
    m1 = vgg16_vfl(NUM_CLASSES).to(device).train()
    head = ClassificationModelHostTrainableHead(hidden_dim=EXPECTED_EMBED_DIM * 2, num_classes=NUM_CLASSES).to(device).train()

    opt = torch.optim.SGD(list(m0.parameters()) + list(m1.parameters()) + list(head.parameters()), lr=0.01)

    x0 = torch.randn(bs, 3, 224, 112, device=device)
    x1 = torch.randn(bs, 3, 224, 112, device=device)
    y = torch.randint(0, NUM_CLASSES, (bs,), device=device)

    opt.zero_grad()
    z0 = m0(x0)
    z1 = m1(x1)

    # shape assertions
    assert z0.dim() == 2 and z1.dim() == 2, f"embeddings must be 2D, got {z0.shape} and {z1.shape}"
    assert z0.shape[1] == EXPECTED_EMBED_DIM, f"expected embed dim {EXPECTED_EMBED_DIM}, got {z0.shape[1]}"
    assert z1.shape[1] == EXPECTED_EMBED_DIM, f"expected embed dim {EXPECTED_EMBED_DIM}, got {z1.shape[1]}"

    out = head([z0, z1])
    assert out.shape[0] == bs and out.shape[1] == NUM_CLASSES, f"head output shape unexpected: {out.shape}"

    loss = F.cross_entropy(out, y)
    loss.backward()
    opt.step()

    # gradients exist
    has_grad_m0 = any(p.grad is not None for p in m0.parameters())
    has_grad_m1 = any(p.grad is not None for p in m1.parameters())
    has_grad_head = any(p.grad is not None for p in head.parameters())

    assert has_grad_m0, "m0 has no gradients after backward"
    assert has_grad_m1, "m1 has no gradients after backward"
    assert has_grad_head, "head has no gradients after backward"

    # simple numeric sanity: head grads non-zero
    head_grad_norm_sq = 0.0
    for p in head.parameters():
        if p.grad is not None:
            head_grad_norm_sq += p.grad.data.norm().item() ** 2
    head_grad_norm = math.sqrt(head_grad_norm_sq)
    assert head_grad_norm > 0.0, "head gradient norm is zero"

    return True


# pytest-style test function
def test_embedding_and_grad_flow_cpu():
    device = torch.device('cpu')
    assert run_check(bs=2, device=device)


if __name__ == '__main__':
    # try CUDA first if available
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running test on', dev)
    run_check(bs=2, device=dev)
    print('ok')
