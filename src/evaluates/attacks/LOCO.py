import os
import sys

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F

from evaluates.attacks.GradientBasedAttackBase import GradientBasedAttackBase


class LOCO(GradientBasedAttackBase):
    def __init__(self, top_vfl, args):
        super().__init__(top_vfl, args)
        self.attack_name = "LOCO"
        self.attack_configs = args.attack_configs or {}
        self.replacement = self.attack_configs.get("replacement", "zero")
        self.attack_on = self.attack_configs.get("attack_on", "test")
        configured_parties = self.attack_configs.get("party", list(range(self.k)))
        self.eval_parties = [int(party_id) for party_id in configured_parties]

        if self.attack_on not in ["test", "train"]:
            raise ValueError("LOCO attack_on must be test or train")
        if self.replacement not in ["zero", "mean"]:
            raise ValueError("LOCO replacement must be zero or mean")

    def _prepare_replacement_values(self, data_list):
        replacement_values = {}
        for party_id, tensor in enumerate(data_list):
            if self.replacement == "zero":
                replacement_values[party_id] = torch.zeros_like(tensor[0:1]).to(self.device).float()
            else:
                replacement_values[party_id] = tensor.mean(dim=0, keepdim=True).to(self.device).float()
        return replacement_values

    def _run_eval_pass(self, data_list, labels, removed_party_id=None, replacement_values=None):
        total = labels.shape[0]
        batch_size = getattr(self.args, "test_batch_size", self.args.batch_size)

        accuracy_correct = 0
        prediction_changed = 0
        total_logit_shift = 0.0
        total_true_prob_drop = 0.0

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                clean_inputs = [data_list[party_id][start:end].to(self.device).float() for party_id in range(self.k)]
                clean_logits, _ = self._forward_with_party_inputs(clean_inputs)
                clean_pred = torch.argmax(clean_logits, dim=-1)
                clean_prob = F.softmax(clean_logits, dim=-1)

                eval_inputs = [tensor.clone() for tensor in clean_inputs]
                if removed_party_id is not None:
                    replacement = replacement_values[removed_party_id].repeat(end - start, *([1] * (eval_inputs[removed_party_id].dim() - 1)))
                    eval_inputs[removed_party_id] = replacement

                eval_logits, _ = self._forward_with_party_inputs(eval_inputs)
                eval_pred = torch.argmax(eval_logits, dim=-1)
                eval_prob = F.softmax(eval_logits, dim=-1)

                batch_labels = labels[start:end].to(self.device)
                accuracy_correct += (eval_pred == batch_labels).sum().item()
                prediction_changed += (eval_pred != clean_pred).sum().item()
                total_logit_shift += (eval_logits - clean_logits).abs().reshape(end - start, -1).mean(dim=1).sum().item()

                clean_true_prob = clean_prob.gather(1, batch_labels.unsqueeze(1)).squeeze(1)
                eval_true_prob = eval_prob.gather(1, batch_labels.unsqueeze(1)).squeeze(1)
                total_true_prob_drop += torch.clamp(clean_true_prob - eval_true_prob, min=0.0).sum().item()

        return {
            "accuracy": accuracy_correct / max(total, 1),
            "prediction_change_rate": prediction_changed / max(total, 1),
            "mean_logit_shift": total_logit_shift / max(total, 1),
            "mean_true_class_prob_drop": total_true_prob_drop / max(total, 1),
        }

    def attack(self):
        for party_id in range(self.k):
            self.top_vfl.parties[party_id].local_model.eval()
        self.top_vfl.parties[self.active_party_id].global_model.eval()

        data_list = [self._get_party_data(party_id, self.attack_on) for party_id in range(self.k)]
        label_tensor = self._label_indices(self._get_party_labels(self.active_party_id, self.attack_on))
        replacement_values = self._prepare_replacement_values(data_list)

        baseline_metrics = self._run_eval_pass(data_list, label_tensor)
        party_results = []

        for party_id in self.eval_parties:
            leave_one_out_metrics = self._run_eval_pass(
                data_list,
                label_tensor,
                removed_party_id=party_id,
                replacement_values=replacement_values,
            )
            accuracy_drop = baseline_metrics["accuracy"] - leave_one_out_metrics["accuracy"]
            party_results.append(
                {
                    "party_id": party_id,
                    "accuracy_without_party": leave_one_out_metrics["accuracy"],
                    "accuracy_drop": accuracy_drop,
                    "prediction_change_rate": leave_one_out_metrics["prediction_change_rate"],
                    "mean_logit_shift": leave_one_out_metrics["mean_logit_shift"],
                    "mean_true_class_prob_drop": leave_one_out_metrics["mean_true_class_prob_drop"],
                }
            )

        positive_total = sum(max(result["accuracy_drop"], 0.0) for result in party_results)
        for result in party_results:
            if positive_total > 0:
                result["contribution_score"] = max(result["accuracy_drop"], 0.0) / positive_total
            else:
                result["contribution_score"] = 0.0

        metrics = {
            "attack_on": self.attack_on,
            "replacement": self.replacement,
            "clean_accuracy": baseline_metrics["accuracy"],
            "party_metrics": party_results,
        }
        print("LOCO metrics:", metrics)
        return metrics
