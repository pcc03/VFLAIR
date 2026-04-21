import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F

from evaluates.attacks.attacker import Attacker
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res


class PGD(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.vfl = top_vfl
        self.device = args.device
        self.k = args.k

        self.eps = args.attack_configs.get("eps", 0.3)
        self.alpha = args.attack_configs.get("alpha", 0.01)
        self.steps = args.attack_configs.get("steps", 40)
        self.random_start = args.attack_configs.get("random_start", 1)
        self.clamp_min = args.attack_configs.get("clamp_min", 0.0)
        self.clamp_max = args.attack_configs.get("clamp_max", 1.0)
        self.attack_on = args.attack_configs.get("attack_on", "test")
        self.party = args.attack_configs.get("party", [self.k - 1])

        self.file_name = "attack_result.txt"
        self.exp_res_dir = f"exp_result/main/{args.dataset}/attack/PGD/"
        self.exp_res_path = ""
        if not os.path.exists(self.exp_res_dir):
            os.makedirs(self.exp_res_dir)
        self.exp_res_path = self.exp_res_dir + self.file_name

    def _get_loaders(self):
        if self.attack_on == "train":
            return [self.vfl.parties[ik].train_loader for ik in range(self.k)]
        return [self.vfl.parties[ik].test_loader for ik in range(self.k)]

    def _compute_logits(self, inputs):
        preds = []
        for ik in range(self.k):
            preds.append(self.vfl.parties[ik].local_model(inputs[ik]))
        return self.vfl.parties[self.k - 1].global_model(preds)

    def _loss(self, logits, labels):
        if labels.dim() > 1:
            return cross_entropy_for_onehot(logits, labels.float())
        return F.cross_entropy(logits, labels)

    def _accuracy(self, logits, labels):
        pred = torch.argmax(logits, dim=-1)
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=-1)
        return (pred == labels).float().mean().item()

    def _project(self, x_adv, x_orig):
        x_adv = torch.max(torch.min(x_adv, x_orig + self.eps), x_orig - self.eps)
        if self.clamp_min is not None and self.clamp_max is not None:
            x_adv = torch.clamp(x_adv, self.clamp_min, self.clamp_max)
        return x_adv

    def _pgd_attack(self, inputs, labels, attack_party):
        x_orig = inputs[attack_party].detach()
        if self.random_start:
            noise = torch.empty_like(x_orig).uniform_(-self.eps, self.eps)
            x_adv = x_orig + noise
            if self.clamp_min is not None and self.clamp_max is not None:
                x_adv = torch.clamp(x_adv, self.clamp_min, self.clamp_max)
        else:
            x_adv = x_orig.clone()

        for _ in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            inputs_adv = list(inputs)
            inputs_adv[attack_party] = x_adv
            logits = self._compute_logits(inputs_adv)
            loss = self._loss(logits, labels)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv + self.alpha * torch.sign(grad)
            x_adv = self._project(x_adv.detach(), x_orig)
        return x_adv.detach()

    def attack(self):
        if self.args.dataset in ["cora"]:
            raise NotImplementedError("PGD attack is not supported for graph datasets.")

        for ik in range(self.k):
            self.vfl.parties[ik].local_model.eval()
        self.vfl.parties[self.k - 1].global_model.eval()

        clean_correct = 0
        adv_correct = 0
        total_samples = 0

        data_loader_list = self._get_loaders()
        for parties_data in zip(*data_loader_list):
            inputs = []
            for ik in range(self.k):
                inputs.append(parties_data[ik][0].to(self.device))
            labels = parties_data[self.k - 1][1].to(self.device)

            with torch.no_grad():
                clean_logits = self._compute_logits(inputs)
                clean_pred = torch.argmax(clean_logits, dim=-1)
                clean_labels = torch.argmax(labels, dim=-1) if labels.dim() > 1 else labels

            adv_inputs = list(inputs)
            for attack_party in self.party:
                adv_inputs[attack_party] = self._pgd_attack(adv_inputs, labels, attack_party)

            with torch.no_grad():
                adv_logits = self._compute_logits(adv_inputs)
                adv_pred = torch.argmax(adv_logits, dim=-1)

            clean_correct += (clean_pred == clean_labels).sum().item()
            adv_correct += (adv_pred == clean_labels).sum().item()
            total_samples += clean_labels.numel()

        clean_acc_avg = clean_correct / max(total_samples, 1)
        adv_acc_avg = adv_correct / max(total_samples, 1)
        acc_drop = clean_acc_avg - adv_acc_avg

        exp_result = (
            f"eps|alpha|steps|attack_on|party|clean_acc|adv_acc|acc_drop,"
            f"{self.eps}|{self.alpha}|{self.steps}|{self.attack_on}|{self.party}|"
            f"{clean_acc_avg:.6f}|{adv_acc_avg:.6f}|{acc_drop:.6f}"
        )
        append_exp_res(self.exp_res_path, exp_result)
        print(exp_result)
        return adv_acc_avg
