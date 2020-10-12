import torch
from torch import nn
import torch.nn.functional as F
import math

class CosLinear(nn.Module):
    def __init__(self, embed_dim: int, num_labels: int,
                 alpha: float = 1.0, learn_alpha: bool = False,
                 label_embeds = None):
        super(CosLinear, self).__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        if not learn_alpha:
            self.alpha.requires_grad = False
        if label_embeds:
            assert label_embeds.shape[0] == num_labels, "label_embeds must have first dimension = num_labels"
            assert label_embeds.shape[1] == embed_dim, "label_embeds must have second dimension = embed_dim"
            self.label_embeds = label_embeds
        else:
            self.label_embeds = nn.Parameter(torch.Tensor(num_labels, embed_dim))
            nn.init.kaiming_uniform_(self.label_embeds, a=math.sqrt(5)) # taken from nn.Linear

    def forward(self, input):
        # input: (bsz x embed_dim)
        input_norm = torch.norm(input, 2, 1) # shape: (bsz)
        label_embeds_norm = torch.norm(self.label_embeds, 2, 1) # shape: (num_labels)
        norm_matrix = torch.matmul(input_norm.unsqueeze(-1),
                                   label_embeds_norm.unsqueeze(0)) # shape: (bsz, num_labels)
        out = F.linear(input, self.label_embeds) # shape: (bsz, num_labels)
        return self.alpha * out / norm_matrix

