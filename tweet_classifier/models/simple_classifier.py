from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from tweet_classifier import L2Linear, CosLinear

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Average, F1Measure

@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier: str = "linear",
                 alpha: float = 1.0,
                 learn_alpha: bool = False,
                 l2_to_sim: str = "negative",
                 squared_l2: bool = False,
                 truncate: bool = False,
                 embeds_per_label: int = 1,
                 label_namespace: str = "labels",
                 ):
        super().__init__(vocab)

        # str arg validation
        valid_classifiers = ["linear", "l2", "cos"]
        assert classifier in valid_classifiers, f"classifier must be in {valid_classifiers}"
        self.classifier = classifier
        valid_l2_to_sim = ["negative", "inverse"]
        assert l2_to_sim in valid_l2_to_sim, f"l2_to_sim must be in {valid_l2_to_sim}"
        self.l2_to_sim = l2_to_sim

        # encoder and embedder layers
        self.embedder = embedder
        self.encoder = encoder
        self.labels = vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self.num_labels = len(self.labels)
        self.embeds_per_label = embeds_per_label
        self.classifier_out = self.num_labels * embeds_per_label
        self.embed_dim = encoder.get_output_dim()

        # similarity/distance layer
        if classifier == "linear":
            self.classifier = nn.Linear(self.embed_dim, self.classifier_out)
        elif classifier == "l2":
            self.classifier = L2Linear(self.embed_dim, self.classifier_out, square=squared_l2)
        elif classifier == "cos":
            self.classifier = CosLinear(self.embed_dim, self.classifier_out)
        else:
            raise ValueError(f"Invalid classifier value: {classifier}")

        # truncate logits
        self.truncate = truncate
        if truncate:
            if classifier == "linear":
                self.threshold = nn.Parameter(torch.Tensor([0.1]))
            elif classifier == "cos":
                self.threshold = nn.Parameter(torch.Tensor([0.1]))
            elif classifier == "l2":
                if l2_to_sim == "negative":
                    self.threshold = nn.Parameter(torch.Tensor([float(self.embed_dim)]))
                elif l2_to_sim == "inverse":
                    self.threshold = nn.Parameter(torch.Tensor([-1.0]))
                else:
                    raise ValueError(f"Invalid l2_to_sim value: {l2_to_sim}")
            else:
                raise ValueError(f"Invalid classifier value: {classifier}")
                # str arg validation
                valid_classifiers = ["linear", "l2", "cos"]
                assert classifier in valid_classifiers, f"classifier must be in {valid_classifiers}"
                self.classifier = classifier
                valid_l2_to_sim = ["negative", "inverse"]
                assert l2_to_sim in valid_l2_to_sim, f"l2_to_sim must be in {valid_l2_to_sim}"
                self.l2_to_sim = l2_to_sim

        # scale logits by alpha
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        if not learn_alpha:
            self.alpha.requires_grad = False

        # metrics
        self.accuracy = CategoricalAccuracy()
        self.prf_metrics = {l: F1Measure(i) for i, l in self.labels.items()}
        self.avg_alpha = Average()
        if self.truncate:
            self.trunc_avg_num = Average()
            self.trunc_avg_untrunc_num = Average()
            self.trunc_avg_threshold = Average()
            self.trunc_avg_sim = Average()
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, classifier_out)
        logits = self.classifier(encoded_text)
        if self.classifier == "l2":
            if self.l2_to_sim == "negative":
                logits = logits.mul(-1.0)
            elif self.l2_to_sim == "inverse":
                logits = logits.pow(-1)
        if self.truncate:
            logits -= self.threshold
            logits = logits.relu()
            if self.embeds_per_label > 1:
                bsz = len(logits)
                d0 = bsz
                d1 = self.classifier_out / self.embeds_per_label
                d2 = self.embeds_per_label
                # Shape: (batch_size, num_labels)
                logits = logits.reshape((d0,d1,d2)).sum(dim=-1)
            if label is not None:
                self.trunc_avg_num(logits[logits==0].numel())
                self.trunc_avg_untrunc_num(logits[logits!=0].numel())
                self.trunc_avg_threshold(self.threshold.item())
                self.trunc_avg_sim(logits[logits!=0].mean().item())
        logits = logits.mul(self.alpha)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            self.avg_alpha(self.alpha.item())
            for metric in self.prf_metrics.values():
                metric(logits, label)
            output['loss'] = F.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(reset)}
        metrics.update({"alpha": self.avg_alpha.get_metric(reset)})
        if self.truncate:
            metrics.update({
                "trunc_avg_num": self.trunc_avg_num.get_metric(reset),
                "trunc_avg_untrunc_num": self.trunc_avg_untrunc_num.get_metric(reset),
                "trunc_avg_threshold": self.trunc_avg_threshold.get_metric(reset),
                "trunc_avg_sim": self.trunc_avg_sim.get_metric(reset),
            })
        # precision/recall/f1
        metrics.update({f"{l}_P": m.get_metric()[0] for l, m in self.prf_metrics.items()})
        metrics.update({f"{l}_R": m.get_metric()[1] for l, m in self.prf_metrics.items()})
        metrics.update({f"{l}_F1": m.get_metric()[2] for l, m in self.prf_metrics.items()})
        return metrics
