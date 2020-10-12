from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

from tweet_classifier import L2Linear, CosLinear

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Average, F1Measure

EMBEDDERS = {
    'bert-base-uncased': BertModel,
    'bert-base-cased': BertModel,
}


@Model.register('simple_attention_classifier')
class SimpleAttentionClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: str,
                 encoder: Seq2VecEncoder,
                 classifier: str = "linear",
                 alpha: float = 1.0,
                 learn_alpha: bool = False,
                 l2_to_sim: str = "negative",
                 squared_l2: bool = False,
                 truncate: bool = False,
                 embeds_per_label: int = 1,
                 label_namespace: str = "labels",
                 attention_layer: str = "first",
                 finetune_bert: bool = True,
                 trunc_ratio: float = 0.1,
                 ):
        super().__init__(vocab)

        # str arg validation
        assert embedder in EMBEDDERS.keys(), f"embedder must be in {list(EMBEDDERS.keys())}"
        valid_classifiers = ["linear", "l2", "cos"]
        assert classifier in valid_classifiers, f"classifier must be in {valid_classifiers}"
        self.classifier_type = classifier
        valid_l2_to_sim = ["negative", "inverse"]
        assert l2_to_sim in valid_l2_to_sim, f"l2_to_sim must be in {valid_l2_to_sim}"
        self.l2_to_sim = l2_to_sim
        self.finetune_bert = finetune_bert
        self.squared_l2 = squared_l2
        self.trunc_ratio = trunc_ratio

        # encoder and embedder layers
        self.embedder = EMBEDDERS[embedder].from_pretrained(embedder, output_attentions=True)
        self.encoder = encoder
        self.labels = vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self.num_labels = len(self.labels)
        self.embeds_per_label = embeds_per_label
        self.classifier_out = self.num_labels * embeds_per_label
        self.embed_dim = encoder.get_output_dim()
        self.attention_layer = attention_layer

        # similarity/distance layer
        if self.classifier_type == "linear":
            self.classifier = nn.Linear(self.embed_dim, self.classifier_out)
        elif self.classifier_type == "l2":
            self.classifier = L2Linear(self.embed_dim, self.classifier_out, square=squared_l2)
        elif self.classifier_type == "cos":
            self.classifier = CosLinear(self.embed_dim, self.classifier_out)
        else:
            raise ValueError(f"Invalid classifier value: {classifier}")

        # truncate logits
        self.truncate = truncate
        if truncate:
            # compute threshold values from a dummy learnable embedding
            # for stable and sufficient gradient updates
            self.trunc_embed = nn.Parameter(torch.ones(self.embed_dim) * 0.5)

        # scale logits by alpha
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        if not learn_alpha:
            self.alpha.requires_grad = False

        # metrics
        self.accuracy = CategoricalAccuracy()
        self.prf_metrics = {l: F1Measure(i) for i, l in self.labels.items()}
        self.avg_alpha = Average()
        if self.truncate:
            self.trunc_avg_total_num = Average()
            self.trunc_avg_trunc_num = Average()
            self.trunc_avg_untrunc_num = Average()
            self.trunc_avg_threshold = Average()
            self.trunc_avg_sim = Average()
            self.trunc_pre_avg_sim = Average()
            self.trunc_avg_sim_std = Average()
            self.trunc_pre_avg_sim_std = Average()
            self.trunc_pre_avg_sim_std = Average()

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if label is None or not self.finetune_bert:
            self.embedder.eval()
        else:
            self.embedder.train()
        # Shape: (batch_size, num_tokens, embedding_dim)
        token_ids = text["bert"]["token_ids"]
        outputs = self.embedder(token_ids)
        embedded_text = outputs[0]
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, classifier_out)
        logits = self.classifier(encoded_text)
        if self.classifier_type == "l2":
            if self.l2_to_sim == "negative":
                logits = logits.mul(-1.0)
            elif self.l2_to_sim == "inverse":
                logits = logits.pow(-1)
        if self.truncate:
            if label is not None:
                with torch.no_grad():
                    self.trunc_pre_avg_sim(logits.mean().item())
                    self.trunc_pre_avg_sim_std(logits.std().item())
                    self.avg_alpha(self.alpha.item())
            if self.classifier_type == "linear":
                threshold = self.trunc_embed.sum()
            elif self.classifier_type == "cos":
                threshold = self.trunc_embed.mean()
            elif self.classifier_type == "l2":
                threshold = (1.5*self.trunc_embed).norm(2)
                if self.squared_l2:
                    threshold = threshold ** 2
                if self.l2_to_sim == "negative":
                    threshold = threshold.mul(-1)
                elif self.l2_to_sim == "inverse":
                    threshold = threshold.pow(-1)
            # weighted sum to prevent instability / nans / 0 gradients
            truncated_logits = (logits - threshold).relu()
            logits = self.trunc_ratio * logits + \
                     (1-self.trunc_ratio) * truncated_logits

            if label is not None:
                self.trunc_avg_total_num(truncated_logits.numel())
                self.trunc_avg_trunc_num(truncated_logits[truncated_logits==0].numel())
                self.trunc_avg_untrunc_num(truncated_logits[truncated_logits!=0].numel())
                self.trunc_avg_threshold(threshold.item())
                self.trunc_avg_sim(truncated_logits[truncated_logits!=0].mean().item())
                self.trunc_avg_sim_std(truncated_logits[truncated_logits!=0].std().item())

        logits = logits.mul(self.alpha)
        # Shape: (batch_size, classifier_out)
        if self.embeds_per_label > 1:
            bsz = len(logits)
            d0 = bsz
            d1 = int(self.classifier_out / self.embeds_per_label)
            d2 = int(self.embeds_per_label)
            # Shape: (batch_size, num_labels)
            logits = logits.reshape((d0, d1, d2)).sum(dim=-1)
        probs = F.softmax(logits, dim=-1)
        output = {'probs': probs.detach().tolist()}
        if label is None:
            # Shape: (num_layers, bsz, num_attn_heads, seq_len)
            attention = torch.stack(outputs[-1])[:,:,:,0,:]
            # Shape: (bsz, num_attn_heads, seq_len)
            if self.attention_layer == "first":
                attention = attention[0]
            elif self.attention_layer == "last":
                attention = attention[-1]
            # Shape: (bsz, seq_len)
            attention = attention.mean(dim=1)
            attention /= attention.sum(dim=1).unsqueeze(-1)

            tokens = []
            for seq in token_ids:
                tokens.append([self.vocab.get_token_from_index(i.item(), "tags") for i in seq])
            output.update({'encoded_text': encoded_text.detach().tolist()})
            output.update({'attention': attention.detach().tolist()})
            output.update({'tokens': tokens})
        else:
            self.accuracy(logits, label)
            for metric in self.prf_metrics.values():
                metric(logits, label)
            output['loss'] = F.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(reset)}
        metrics.update({"alpha": self.avg_alpha.get_metric(reset)})
        if self.truncate:
            metrics.update({
                "trunc_avg_total_num": self.trunc_avg_total_num.get_metric(reset),
                "trunc_avg_trunc_num": self.trunc_avg_trunc_num.get_metric(reset),
                "trunc_avg_untrunc_num": self.trunc_avg_untrunc_num.get_metric(reset),
                "trunc_avg_threshold": self.trunc_avg_threshold.get_metric(reset),
                "trunc_avg_sim": self.trunc_avg_sim.get_metric(reset),
                "trunc_pre_avg_sim": self.trunc_pre_avg_sim.get_metric(reset),
                "trunc_avg_sim_std": self.trunc_avg_sim_std.get_metric(reset),
                "trunc_pre_avg_sim_std": self.trunc_pre_avg_sim_std.get_metric(reset),
            })
        # precision/recall/f1
        metrics.update({f"{l}_P": m.get_metric()[0] for l, m in self.prf_metrics.items()})
        metrics.update({f"{l}_R": m.get_metric()[1] for l, m in self.prf_metrics.items()})
        metrics.update({f"{l}_F1": m.get_metric()[2] for l, m in self.prf_metrics.items()})

        if self.classifier_type == "l2":
            self._get_class_l2_mean_and_std()
            metrics["class_l2"] = self.class_l2_mean
            metrics["class_l2_std"] = self.class_l2_std
        return metrics

    def _get_class_l2_mean_and_std(self):
        with torch.no_grad():
            class_l2 = self.classifier(self.classifier.label_embeds.t())
            class_l2 = torch.triu(class_l2, diagonal=1)
            class_l2 = class_l2[class_l2 != 0]
            self.class_l2_mean = class_l2.mean().item()
            self.class_l2_std = class_l2.std().item()
