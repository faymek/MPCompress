import math
import torch.nn.functional as F
import discrete_entropy.ops as ops

class Softmax:
    def __init__(self, logits):
        super().__init__()
        self.batch_shape = logits.shape[:-1]
        self.pmf_length = logits.shape[-1]
        self.logits = logits

    def prob(self, indexes):
        pmf = F.softmax(self.logits, dim=-1)
        prob = pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return ops.lower_bound(prob, 1e-9)
        else:
            return prob

    def log_prob(self, indexes):
        log_pmf = F.log_softmax(self.logits, dim=-1)
        log_prob = log_pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return ops.lower_bound(log_prob, math.log(1e-9))
        else:
            return log_prob

    def pmf(self):
        pmf = F.softmax(self.logits, dim=-1)

        if pmf.requires_grad:
            return ops.lower_bound(pmf, 1e-9)
        else:
            return pmf

    def log_pmf(self):
        log_pmf = F.log_softmax(self.logits, dim=-1)

        if log_pmf.requires_grad:
            return ops.lower_bound(log_pmf, math.log(1e-9))
        else:
            return log_pmf