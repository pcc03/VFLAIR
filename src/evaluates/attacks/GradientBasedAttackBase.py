import os
import sys

sys.path.append(os.pardir)

import numpy as np
import torch

from evaluates.attacks.attacker import Attacker


class GradientBasedAttackBase(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.top_vfl = top_vfl
        self.device = args.device
        self.k = args.k
        self.num_classes = args.num_classes
        self.active_party_id = args.k - 1

    def set_seed(self, seed=0):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _require_tensor_party_data(self, party_id, split):
        party = self.top_vfl.parties[party_id]
        data = getattr(party, f"{split}_data")
        if not isinstance(data, torch.Tensor):
            raise NotImplementedError(
                f"{self.__class__.__name__} currently supports tensor {split}_data only, "
                f"but party {party_id} has type {type(data)}"
            )
        return data

    def _get_party_data(self, party_id, split):
        return self._require_tensor_party_data(party_id, split)

    def _get_party_labels(self, party_id, split):
        labels = getattr(self.top_vfl.parties[party_id], f"{split}_label")
        if not isinstance(labels, torch.Tensor):
            raise NotImplementedError(
                f"{self.__class__.__name__} currently supports tensor {split}_label only, "
                f"but party {party_id} has type {type(labels)}"
            )
        return labels

    def _label_indices(self, labels):
        if labels.dim() > 1:
            return torch.argmax(labels, dim=-1).long()
        return labels.long()

    def _batch_to_device(self, tensor, requires_grad=False):
        batch = tensor.detach().clone().to(self.device).float()
        if requires_grad:
            batch.requires_grad_(True)
        return batch

    def _forward_with_party_inputs(self, party_inputs):
        pred_list = []
        for party_id in range(self.k):
            pred_list.append(self.top_vfl.parties[party_id].local_model(party_inputs[party_id]))
        logits = self.top_vfl.parties[self.active_party_id].global_model(pred_list)
        return logits, pred_list

    def _evaluate_clean_accuracy(self):
        for party_id in range(self.k):
            self.top_vfl.parties[party_id].local_model.eval()
        self.top_vfl.parties[self.active_party_id].global_model.eval()

        data_list = [self._get_party_data(party_id, "test") for party_id in range(self.k)]
        label_tensor = self._label_indices(self._get_party_labels(self.active_party_id, "test"))
        total = label_tensor.shape[0]
        correct = 0
        batch_size = getattr(self.args, "test_batch_size", self.args.batch_size)

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                party_inputs = [
                    data_list[party_id][start:end].to(self.device).float() for party_id in range(self.k)
                ]
                logits, _ = self._forward_with_party_inputs(party_inputs)
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == label_tensor[start:end].to(self.device)).sum().item()

        return correct / max(total, 1)
