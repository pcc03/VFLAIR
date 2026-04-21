import argparse
import os
import types

import torch

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
os.chdir(SRC_DIR)
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from load.LoadParty import load_parties
from evaluates.attacks.PGD import PGD


def parse_args():
    parser = argparse.ArgumentParser(description="Validate clean and PGD accuracy from a saved VFL checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pth checkpoint.")
    parser.add_argument("--eps", type=float, default=0.03, help="PGD epsilon.")
    parser.add_argument("--alpha", type=float, default=0.0078, help="PGD step size.")
    parser.add_argument("--steps", type=int, default=20, help="PGD steps.")
    parser.add_argument("--party", type=int, nargs="+", default=[1], help="Party index or indices to attack.")
    parser.add_argument("--attack-on", default="test", choices=["train", "test"], help="Split to attack.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for a faster smoke test.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override device. Defaults to cuda when available, otherwise cpu.",
    )
    return parser.parse_args()


def label_index(labels):
    return torch.argmax(labels, dim=-1) if labels.dim() > 1 else labels


def build_runtime_args(checkpoint_args, cli_args):
    args = types.SimpleNamespace(**checkpoint_args)
    args.device = cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.need_auxiliary = 0
    args.parties = None
    args.apply_backdoor = False
    args.apply_ns = False
    args.apply_mf = False
    args.apply_nl = False
    args.apply_cae = False
    args.attack_name = "PGD"
    args.attack_configs = {
        "eps": cli_args.eps,
        "alpha": cli_args.alpha,
        "steps": cli_args.steps,
        "random_start": 1,
        "clamp_min": 0.0,
        "clamp_max": 1.0,
        "attack_on": cli_args.attack_on,
        "party": cli_args.party,
    }
    return args


def load_checkpoint_models(args, checkpoint):
    args = load_parties(args)
    for ik in range(args.k):
        args.parties[ik].prepare_data_loader(batch_size=args.batch_size)
        args.parties[ik].local_model.load_state_dict(checkpoint["model_state_dicts"][ik])
        args.parties[ik].local_model.to(args.device).eval()
    args.parties[args.k - 1].global_model.load_state_dict(checkpoint["model_state_dicts"][args.k])
    args.parties[args.k - 1].global_model.to(args.device).eval()
    return args


def evaluate_clean_and_pgd(args, max_batches=None):
    pgd = PGD(types.SimpleNamespace(parties=args.parties), args)
    clean_correct = 0
    adv_correct = 0
    total_samples = 0
    clean_batch_acc_sum = 0.0
    adv_batch_acc_sum = 0.0
    total_batches = 0

    data_loader_list = [party.train_loader if args.attack_configs["attack_on"] == "train" else party.test_loader
                        for party in args.parties]

    for batch_idx, parties_data in enumerate(zip(*data_loader_list)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = [parties_data[ik][0].to(args.device) for ik in range(args.k)]
        labels_onehot = parties_data[args.k - 1][1].to(args.device)
        labels = label_index(labels_onehot)

        with torch.no_grad():
            clean_logits = pgd._compute_logits(inputs)
        clean_pred = torch.argmax(clean_logits, dim=-1)
        clean_correct += (clean_pred == labels).sum().item()
        clean_batch_acc_sum += (clean_pred == labels).float().mean().item()

        adv_inputs = list(inputs)
        for attack_party in args.attack_configs["party"]:
            adv_inputs[attack_party] = pgd._pgd_attack(adv_inputs, labels_onehot, attack_party)

        with torch.no_grad():
            adv_logits = pgd._compute_logits(adv_inputs)
        adv_pred = torch.argmax(adv_logits, dim=-1)
        adv_correct += (adv_pred == labels).sum().item()
        adv_batch_acc_sum += (adv_pred == labels).float().mean().item()

        total_samples += labels.numel()
        total_batches += 1

        print(
            f"batch={batch_idx} clean_batch_acc={(clean_pred == labels).float().mean().item():.6f} "
            f"adv_batch_acc={(adv_pred == labels).float().mean().item():.6f}"
        )

    return {
        "clean_exact_acc": clean_correct / max(total_samples, 1),
        "adv_exact_acc": adv_correct / max(total_samples, 1),
        "clean_batch_avg_acc": clean_batch_acc_sum / max(total_batches, 1),
        "adv_batch_avg_acc": adv_batch_acc_sum / max(total_batches, 1),
        "acc_drop_exact": (clean_correct - adv_correct) / max(total_samples, 1),
        "total_samples": total_samples,
        "total_batches": total_batches,
    }


def main():
    cli_args = parse_args()
    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu")
    args = build_runtime_args(checkpoint["args"], cli_args)
    args = load_checkpoint_models(args, checkpoint)

    print(f"device={args.device}")
    print(f"dataset={args.dataset} k={args.k} batch_size={args.batch_size}")
    print(
        f"attack=PGD eps={args.attack_configs['eps']} alpha={args.attack_configs['alpha']} "
        f"steps={args.attack_configs['steps']} party={args.attack_configs['party']}"
    )

    metrics = evaluate_clean_and_pgd(args, max_batches=cli_args.max_batches)
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
