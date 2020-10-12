import torch
from torch import nn
import math

class L2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeds, label_embeds):
        # embeddings: bsz x embed_dim
        # label_embeds: embed_dim x num_labels
        bsz, embed_dim = embeds.shape
        embed_dim2, num_labels = label_embeds.shape
        assert embed_dim == embed_dim2, "Inner dimensions must match\n" \
                                        f"Embeds: {bsz}x{embed_dim}" \
                                        f"Label Embeds: {embed_dim2}x{num_labels}"
        ctx.save_for_backward(embeds, label_embeds)
        # X = embeds, W = label_embeds, Y = output
        # i = bsz, j = embed_dim, k = num_labels
        # Y = sum_{j}^{J} [X_{i,j}^2 - 2(X_{i,j}*W_{j,k}) + W_{j,k}^2]
        #                      (1)            (2)               (3)
        out = (-2 * torch.mm(embeds, label_embeds)     # (2)  // (i,k)
              + (embeds**2).sum(dim=-1).unsqueeze(-1)  # (1) // +(i,1) broadcast
              + (label_embeds**2).sum(dim=0))          # (3) // +(k,) broadcast
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # embeds: bsz x embed_dim
        # label_embeds: embed_dim x num_labels
        # grad_output: bsz x num_labels
        embeds, label_embeds, = ctx.saved_tensors
        # X = embeds, W = label_embeds, dL/dY = grad_output
        # i = bsz, j = embed_dim, k = num_labels
        # dL/dX{i,j} = 2 * sum_{k}^{num_labels} [dL/dY_{i,k} * (X_{i,j} - W_{j,k})]
        #                                                         (1)        (2)
        # dL/dW{j,k} = 2 * sum_{i}^{bsz} [dL/dY_{i,k} * (W_{j,k}) - X_{i,j}]
        #                                                   (3)        (4)
        d_embed = 2 * (embeds * grad_output.sum(dim=-1).unsqueeze(-1) # (1) // (i,j)*(i,1) broadcast
                       - torch.mm(grad_output, label_embeds.t()))     # (2) // (i,j)x(j,k) matmul
        d_label_embeds = 2 * (label_embeds * grad_output.sum(dim=0)   # (3) // (j,k)*(k) broadcast
                       - torch.mm(embeds.t(), grad_output))           # (4) // (j,i)x(i,k) matmul
        return d_embed, d_label_embeds

class L2Linear(nn.Module):
    def __init__(self, embed_dim: int, num_labels: int, square: bool = False,
                 alpha: float = 1.0, learn_alpha: bool = False,
                 negative_l2: bool = False,
                 inverse_l2: bool = False,
                 truncate: bool = False,
                 kaiming_uniform: bool = False,
                 label_embeds = None):
        super(L2Linear, self).__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.square = square
        self.kaiming_uniform = kaiming_uniform

        if label_embeds:
            assert label_embeds.shape[0] == embed_dim, "label_embeds must have first dimension = embed_dim"
            assert label_embeds.shape[1] == num_labels, "label_embeds must have second dimension = num_labels"
            self.label_embeds = label_embeds
        else:
            self.label_embeds = nn.Parameter(torch.Tensor(embed_dim, num_labels).uniform_(-1,1))
            if self.kaiming_uniform:
                nn.init.kaiming_uniform_(self.label_embeds, a=math.sqrt(5)) # taken from nn.Linear


    def forward(self, input):
        out = L2Function.apply(input, self.label_embeds)
        if not self.square:
            out = out.pow(0.5)
        return out
