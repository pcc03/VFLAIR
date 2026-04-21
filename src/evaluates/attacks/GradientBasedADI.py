import os
import sys

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F

from evaluates.attacks.GradientBasedAttackBase import GradientBasedAttackBase


class GradientBasedADI(GradientBasedAttackBase):
    def __init__(self, top_vfl, args):
        super().__init__(top_vfl, args)
        self.attack_name = "GradientBasedADI"
        self.attack_configs = args.attack_configs or {}

        attacker_party = self.attack_configs.get("attacker_party_id", None)
        if attacker_party is None:
            attacker_party = self.attack_configs.get("party", [0])[0]
        self.attacker_party_id = int(attacker_party)

        victim_party = self.attack_configs.get("victim_party_id", None)
        if victim_party is None:
            victim_party = 1 - self.attacker_party_id
        self.victim_party_id = int(victim_party)

        self.strategy = self.attack_configs.get("mutation_mode", "random_mutation")
        self.target_label = int(self.attack_configs.get("target_label", 0))
        self.victim_sample_size = int(self.attack_configs.get("victim_sample_size", 20))
        self.attack_sample_num = int(
            self.attack_configs.get("attack_sample_num", self.attack_configs.get("seed_sample_size", 100))
        )

        # Algorithm 1 parameters.
        self.domination_threshold = float(
            self.attack_configs.get("domination_threshold", self.attack_configs.get("x_percentage", 0.95))
        )
        self.max_rounds = int(
            self.attack_configs.get("max_rounds", self.attack_configs.get("attack_steps", 20))
        )
        self.alpha = float(self.attack_configs.get("alpha", 1.0))
        self.beta = float(self.attack_configs.get("beta", 1.0))
        self.gamma = float(self.attack_configs.get("gamma", 0.1))
        self.sigma = float(self.attack_configs.get("sigma", self.attack_configs.get("momentum", 0.9)))
        self.delta_lr = float(self.attack_configs.get("delta_lr", self.attack_configs.get("lr", 0.01)))
        self.delta_optim_steps = int(self.attack_configs.get("delta_optim_steps", 1))

        # Mutation constraint Lambda is derived from training variance in the bounded strategy.
        self.bound_scale = float(
            self.attack_configs.get(
                "bound_scale",
                self.attack_configs.get("total_mutation_constraint", self.attack_configs.get("lambda", 1.0)),
            )
        )
        self.random_mutation_std = float(self.attack_configs.get("random_mutation_std", 1e-3))
        self.progress_every = int(self.attack_configs.get("progress_every", 50))
        self.seed = int(getattr(args, "current_seed", 0))

        if self.k != 2:
            raise NotImplementedError("GradientBasedADI currently supports two-party VFL only")
        if self.attacker_party_id == self.victim_party_id:
            raise ValueError("attacker_party_id and victim_party_id must be different")
        if sorted([self.attacker_party_id, self.victim_party_id]) != [0, 1]:
            raise ValueError("GradientBasedADI currently supports party ids 0 and 1 only")
        if self.strategy not in ["random_mutation", "bounded_mutation"]:
            raise ValueError("mutation_mode must be random_mutation or bounded_mutation")
        if self.target_label < 0 or self.target_label >= self.num_classes:
            raise ValueError(f"target_label must be in [0, {self.num_classes - 1}]")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")

    def _prepare_attacker_statistics(self, train_tensor):
        train_tensor = train_tensor.float()
        self.train_min = train_tensor.amin(dim=0).to(self.device)
        self.train_max = train_tensor.amax(dim=0).to(self.device)
        variance = train_tensor.var(dim=0, unbiased=False)
        self.lambda_bound = (variance + 1e-12).sqrt().to(self.device) * self.bound_scale

    def _clip_to_data_range(self, adv_input):
        return torch.max(
            torch.min(adv_input, self.train_max.unsqueeze(0)),
            self.train_min.unsqueeze(0),
        )

    def _project_total_mutation(self, clean_input, total_mutation):
        if self.strategy == "bounded_mutation":
            total_mutation = torch.max(
                torch.min(total_mutation, self.lambda_bound.unsqueeze(0)),
                -self.lambda_bound.unsqueeze(0),
            )
        adv_input = self._clip_to_data_range(clean_input + total_mutation)
        return adv_input - clean_input

    def _build_party_inputs(self, attacker_input, victim_input):
        party_inputs = [None] * self.k
        party_inputs[self.attacker_party_id] = attacker_input
        party_inputs[self.victim_party_id] = victim_input
        return party_inputs

    def _loss(self, attacker_input, victim_input):
        logits, _ = self._forward_with_party_inputs(self._build_party_inputs(attacker_input, victim_input))
        target_labels = torch.full(
            (victim_input.shape[0],),
            self.target_label,
            dtype=torch.long,
            device=self.device,
        )
        return F.cross_entropy(logits, target_labels), logits

    def _saliency_est(self, attacker_input, victim_input):
        victim_input = victim_input.detach().clone().requires_grad_(True)
        _, logits = self._loss(attacker_input, victim_input)
        output_variance = torch.var(logits, dim=-1, unbiased=False).mean()
        saliency_grad = torch.autograd.grad(
            output_variance,
            victim_input,
            retain_graph=True,
            create_graph=True,
        )[0]
        saliency = saliency_grad.abs().reshape(saliency_grad.shape[0], -1).sum(dim=1).mean()
        return saliency

    def _objective(self, clean_input, total_mutation, delta, victim_input):
        attacker_input = clean_input + total_mutation + delta
        attacker_input = self._clip_to_data_range(attacker_input)
        saliency = self._saliency_est(attacker_input, victim_input)
        target_loss, _ = self._loss(attacker_input, victim_input)
        objective = self.alpha * saliency + self.beta * target_loss
        if self.strategy == "bounded_mutation":
            objective = objective + self.gamma * torch.norm(delta.reshape(delta.shape[0], -1), p=2, dim=1).mean()
        return objective, saliency.detach(), target_loss.detach()

    def _initialize_delta(self, clean_input):
        if self.strategy == "random_mutation":
            return torch.randn_like(clean_input) * self.random_mutation_std
        return torch.zeros_like(clean_input)

    def _solve_delta_t(self, clean_input, total_mutation, victim_input):
        delta = self._initialize_delta(clean_input)
        for _ in range(self.delta_optim_steps):
            delta_var = delta.detach().clone().requires_grad_(True)
            objective, _, _ = self._objective(clean_input, total_mutation, delta_var, victim_input)
            grad = torch.autograd.grad(objective, delta_var)[0]
            delta = delta_var.detach() - self.delta_lr * grad

            if self.strategy == "bounded_mutation":
                projected_total = self._project_total_mutation(clean_input, total_mutation + delta)
                delta = projected_total - total_mutation
            else:
                attacker_input = self._clip_to_data_range(clean_input + total_mutation + delta)
                delta = attacker_input - clean_input - total_mutation
        return delta.detach()

    def _dominated_proportion(self, attacker_input, victim_set):
        with torch.no_grad():
            repeated_attacker = attacker_input.repeat(victim_set.shape[0], *([1] * (attacker_input.dim() - 1)))
            logits, _ = self._forward_with_party_inputs(
                self._build_party_inputs(repeated_attacker, victim_set.to(self.device).float())
            )
            predictions = torch.argmax(logits, dim=-1)
            return (predictions == self.target_label).float().mean().item()

    def _evaluate_original_pair(self, attacker_input, victim_input, true_label):
        with torch.no_grad():
            logits, _ = self._forward_with_party_inputs(
                self._build_party_inputs(attacker_input, victim_input.to(self.device).float())
            )
            prediction = torch.argmax(logits, dim=-1)
            attacked_accuracy = (prediction == true_label.to(self.device)).float().mean().item()
            target_success = (prediction == self.target_label).float().mean().item()
        return attacked_accuracy, target_success

    def _adi_generation(self, clean_input, victim_set):
        total_mutation = torch.zeros_like(clean_input)
        previous_delta = torch.zeros_like(clean_input)
        round_index = 1
        domination = self._dominated_proportion(clean_input, victim_set)

        while domination < self.domination_threshold and round_index <= self.max_rounds:
            for victim_idx in range(victim_set.shape[0]):
                if domination >= self.domination_threshold or round_index > self.max_rounds:
                    break

                victim_input = victim_set[victim_idx:victim_idx + 1].to(self.device).float()
                delta_t = self._solve_delta_t(clean_input, total_mutation, victim_input)
                delta_t = self.sigma * previous_delta + delta_t

                if self.strategy == "bounded_mutation":
                    new_total_mutation = self._project_total_mutation(clean_input, total_mutation + delta_t)
                    delta_t = new_total_mutation - total_mutation
                    total_mutation = new_total_mutation
                else:
                    total_mutation = self._clip_to_data_range(clean_input + total_mutation + delta_t) - clean_input

                previous_delta = delta_t.detach()
                round_index += 1
                domination = self._dominated_proportion(clean_input + total_mutation, victim_set)

        adv_input = self._clip_to_data_range(clean_input + total_mutation)
        return adv_input.detach(), domination, round_index - 1

    def attack(self):
        self.set_seed(self.seed)

        for party_id in range(self.k):
            self.top_vfl.parties[party_id].local_model.eval()
        self.top_vfl.parties[self.active_party_id].global_model.eval()

        attacker_train = self._get_party_data(self.attacker_party_id, "train")
        attacker_test = self._get_party_data(self.attacker_party_id, "test")
        victim_test = self._get_party_data(self.victim_party_id, "test")
        label_tensor = self._label_indices(self._get_party_labels(self.active_party_id, "test"))

        self._prepare_attacker_statistics(attacker_train)

        clean_accuracy = self._evaluate_clean_accuracy()

        total_test = attacker_test.shape[0]
        attack_sample_num = min(self.attack_sample_num, total_test)
        victim_sample_size = min(self.victim_sample_size, victim_test.shape[0])
        generator = torch.Generator().manual_seed(self.seed)
        seed_indices = torch.randperm(total_test, generator=generator)[:attack_sample_num]
        victim_indices = torch.randperm(victim_test.shape[0], generator=generator)[:victim_sample_size]
        victim_set = victim_test[victim_indices].to(self.device).float()

        domination_scores = []
        clean_domination_scores = []
        attacked_accuracy_total = 0.0
        target_success_total = 0.0
        rounds_used = []

        for seed_index in seed_indices.tolist():
            clean_input = attacker_test[seed_index:seed_index + 1].to(self.device).float()
            clean_domination = self._dominated_proportion(clean_input, victim_set)
            clean_domination_scores.append(clean_domination)
            adv_input, domination, used_rounds = self._adi_generation(clean_input, victim_set)
            domination_scores.append(domination)
            rounds_used.append(used_rounds)

            original_victim = victim_test[seed_index:seed_index + 1]
            true_label = label_tensor[seed_index:seed_index + 1]
            attacked_accuracy, target_success = self._evaluate_original_pair(adv_input, original_victim, true_label)
            attacked_accuracy_total += attacked_accuracy
            target_success_total += target_success

            processed = len(domination_scores)
            if self.progress_every > 0 and (
                processed == 1 or processed % self.progress_every == 0 or processed == attack_sample_num
            ):
                print(
                    f"GradientBasedADI progress: {processed}/{attack_sample_num} seeds, "
                    f"latest_clean_dom={clean_domination:.4f}, latest_adv_dom={domination:.4f}, "
                    f"latest_rounds={used_rounds}"
                )

        sample_count = max(len(seed_indices), 1)
        clean_domination_count_95 = sum(score >= 0.95 for score in clean_domination_scores)
        clean_domination_count_99 = sum(score >= 0.99 for score in clean_domination_scores)
        metrics = {
            "clean_accuracy": clean_accuracy,
            "clean_input_avg_domination_percentage": sum(clean_domination_scores) / max(len(clean_domination_scores), 1),
            "clean_input_domination_success_rate@95%": clean_domination_count_95 / sample_count,
            "clean_input_domination_success_rate@99%": clean_domination_count_99 / sample_count,
            "clean_input_domination_count@95%": clean_domination_count_95,
            "clean_input_domination_count@99%": clean_domination_count_99,
            "attacked_accuracy": attacked_accuracy_total / sample_count,
            "target_success_rate": target_success_total / sample_count,
            "avg_domination_percentage": sum(domination_scores) / max(len(domination_scores), 1),
            "domination_success_rate@95%": sum(score >= 0.95 for score in domination_scores) / sample_count,
            "domination_success_rate@99%": sum(score >= 0.99 for score in domination_scores) / sample_count,
            "domination_threshold": self.domination_threshold,
            "avg_rounds_used": sum(rounds_used) / max(len(rounds_used), 1),
            "victim_sample_size": victim_sample_size,
            "attack_sample_num": sample_count,
            "attacker_party_id": self.attacker_party_id,
            "victim_party_id": self.victim_party_id,
            "target_label": self.target_label,
            "mutation_mode": self.strategy,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "sigma": self.sigma,
        }
        print("GradientBasedADI metrics:", metrics)
        return metrics
