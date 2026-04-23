import os
import sys

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F

from evaluates.attacks.attacker import Attacker
from utils.basic_functions import append_exp_res


class TargetedInferencePerturbation(Attacker):
    """
    Optimize a perturbation delta on party A's input during inference:
      min_delta L(G_full(f_A(x_A+delta), f_B(x_B)), y_t)
               + lambda_a * L(G_A(f_A(x_A+delta)), y_t)
               - lambda_b * L(G_B(f_B(x_B)), y_t)
    """

    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.vfl = top_vfl
        self.device = args.device
        self.k = args.k
        if self.k != 2:
            raise NotImplementedError("TargetedInferencePerturbation currently supports k=2 only.")

        cfg = args.attack_configs if args.attack_configs is not None else {}
        self.eps = float(cfg.get("eps", 0.03))
        self.alpha = float(cfg.get("alpha", 0.005))
        self.steps = int(cfg.get("steps", 20))
        self.random_start = int(cfg.get("random_start", 1))
        self.clamp_min = cfg.get("clamp_min", 0.0)
        self.clamp_max = cfg.get("clamp_max", 1.0)
        self.attack_on = cfg.get("attack_on", "test")
        self.target_label = int(cfg.get("target_label", 0))
        self.lambda_a = float(cfg.get("lambda_a", 1.0))
        self.lambda_b = float(cfg.get("lambda_b", 1.0))
        self.attack_party = int(cfg.get("attack_party", 0))
        # Optional subset controls for faster/safer inference attack runs.
        # <=0 means "use all test batches/samples".
        self.max_test_batches = int(cfg.get("max_test_batches", 0))
        self.max_test_samples = int(cfg.get("max_test_samples", 0))
        if self.attack_party not in [0, 1]:
            raise ValueError("attack_party must be 0 or 1 for two-party VFL.")
        self.other_party = 1 - self.attack_party

        self.file_name = "attack_result.txt"
        self.exp_res_dir = f"exp_result/main/{args.dataset}/attack/TargetedInferencePerturbation/"
        if not os.path.exists(self.exp_res_dir):
            os.makedirs(self.exp_res_dir)
        self.exp_res_path = self.exp_res_dir + self.file_name

    def _get_loaders(self):
        if self.attack_on == "train":
            return [self.vfl.parties[ik].train_loader for ik in range(self.k)]
        return [self.vfl.parties[ik].test_loader for ik in range(self.k)]

    def _ce(self, logits, labels):
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=-1)
        return F.cross_entropy(logits, labels)

    def _project(self, x_adv, x_orig):
        x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
        if self.clamp_min is not None and self.clamp_max is not None:
            x_adv = torch.clamp(x_adv, self.clamp_min, self.clamp_max)
        return x_adv

    def _build_target(self, labels):
        batch = labels.shape[0]
        return torch.full((batch,), self.target_label, dtype=torch.long, device=self.device)

    def _compute_logits(self, x_a, x_b):
        z_a = self.vfl.parties[0].local_model(x_a)
        z_b = self.vfl.parties[1].local_model(x_b)
        logits_full = self.vfl.parties[self.k - 1].global_model([z_a, z_b])
        logits_a = self.vfl.parties[self.k - 1].global_model([z_a, torch.zeros_like(z_b)])
        logits_b = self.vfl.parties[self.k - 1].global_model([torch.zeros_like(z_a), z_b])
        return logits_full, logits_a, logits_b

    def _optimize_delta(self, inputs, labels):
        x_attack_orig = inputs[self.attack_party].detach()
        x_other = inputs[self.other_party].detach()
        target = self._build_target(labels)

        if self.random_start:
            delta = torch.empty_like(x_attack_orig).uniform_(-self.eps, self.eps)
            x_adv = x_attack_orig + delta
            if self.clamp_min is not None and self.clamp_max is not None:
                x_adv = torch.clamp(x_adv, self.clamp_min, self.clamp_max)
        else:
            x_adv = x_attack_orig.clone()

        for _ in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            x_a, x_b = (x_adv, x_other) if self.attack_party == 0 else (x_other, x_adv)
            logits_full, logits_a, logits_b = self._compute_logits(x_a, x_b)

            loss = (
                self._ce(logits_full, target)
                + self.lambda_a * self._ce(logits_a, target)
                - self.lambda_b * self._ce(logits_b, target)
            )
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            # Gradient descent on objective
            x_adv = x_adv - self.alpha * torch.sign(grad)
            x_adv = self._project(x_adv.detach(), x_attack_orig)

        return x_adv.detach()

    def attack(self):
        for ik in range(self.k):
            self.vfl.parties[ik].local_model.eval()
        self.vfl.parties[self.k - 1].global_model.eval()

        total = 0
        clean_target_hits = 0
        adv_target_hits = 0

        data_loader_list = self._get_loaders()
        batch_idx = 0
        for parties_data in zip(*data_loader_list):
            if self.max_test_batches > 0 and batch_idx >= self.max_test_batches:
                break
            inputs = [parties_data[ik][0].to(self.device) for ik in range(self.k)]
            labels = parties_data[self.k - 1][1].to(self.device)
            target = self._build_target(labels)

            with torch.no_grad():
                clean_logits_full, _, _ = self._compute_logits(inputs[0], inputs[1])
                clean_pred = torch.argmax(clean_logits_full, dim=-1)

            adv_inputs = list(inputs)
            adv_inputs[self.attack_party] = self._optimize_delta(inputs, labels)

            with torch.no_grad():
                adv_logits_full, _, _ = self._compute_logits(adv_inputs[0], adv_inputs[1])
                adv_pred = torch.argmax(adv_logits_full, dim=-1)

            clean_target_hits += (clean_pred == target).sum().item()
            adv_target_hits += (adv_pred == target).sum().item()
            total += target.numel()
            batch_idx += 1
            if self.max_test_samples > 0 and total >= self.max_test_samples:
                break

        clean_target_rate = clean_target_hits / max(total, 1)
        target_success_rate = adv_target_hits / max(total, 1)

        exp_result = (
            f"eps|alpha|steps|attack_on|attack_party|target_label|lambda_a|lambda_b|"
            f"clean_target_rate|target_success_rate,"
            f"{self.eps}|{self.alpha}|{self.steps}|{self.attack_on}|{self.attack_party}|"
            f"{self.target_label}|{self.lambda_a}|{self.lambda_b}|"
            f"{clean_target_rate:.6f}|{target_success_rate:.6f}"
        )
        append_exp_res(self.exp_res_path, exp_result)
        print(exp_result)
        return target_success_rate
