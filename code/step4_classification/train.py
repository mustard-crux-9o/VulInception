"""Main training/inference entry for the dual-branch membership classifier."""
import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from embedding import CodeBERTEmbedder
from losses import CELoss, FocalLoss, ASLLoss
from dataset import (
    ClassifierDataLoader, ClassifierDataset, classifier_collate_fn,
    standardize_feature_lists,
)
from model import DualBranchModel


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except TypeError:
        pass


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _get_loss_fn(loss_type, device):
    if loss_type == "ce":
        return CELoss().to(device)
    if loss_type == "focal":
        return FocalLoss().to(device)
    if loss_type == "asl":
        return ASLLoss().to(device)
    raise ValueError(f"Unknown loss type: {loss_type}")


def train_model(model, train_loader, val_loader, epochs, lr, device, loss_type="ce", alpha=0.5, save_path=None):
    """Train dual-branch model, returning best model and metrics."""
    loss_fn = _get_loss_fn(loss_type, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1 = 0.0
    best_metrics = {}
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for prob_stats, code_pooled, labels in train_loader:
            prob_stats, code_pooled, labels = prob_stats.to(device), code_pooled.to(device), labels.to(device)
            optimizer.zero_grad()
            lp = model.forward_prob(prob_stats)
            lc = model.forward_code(code_pooled)
            loss = alpha * loss_fn(lp, labels.long()) + (1 - alpha) * loss_fn(lc, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        model.eval()
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for prob_stats, code_pooled, labels in val_loader:
                prob_stats, code_pooled, labels = prob_stats.to(device), code_pooled.to(device), labels.to(device)
                pp = torch.softmax(model.forward_prob(prob_stats), dim=-1)[:, 1]
                pc = torch.softmax(model.forward_code(code_pooled), dim=-1)[:, 1]
                preds = ((alpha * pp + (1 - alpha) * pc) >= 0.5).float()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        acc = (tp + tn) / (tp + fp + tn + fn + 1e-8)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | acc {acc:.4f} | prec {prec:.4f} | rec {rec:.4f} | f1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"epoch": epoch, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                            "acc": acc, "precision": prec, "recall": rec, "f1": f1}
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best F1: {f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            torch.save(best_state, save_path)
    return model, best_metrics


def inference_and_save(model, output_path, test_loader, records, device, alpha=0.5):
    """Run inference and optionally save predictions to JSONL."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for prob_stats, code_pooled, labels in test_loader:
            prob_stats, code_pooled = prob_stats.to(device), code_pooled.to(device)
            pp = torch.softmax(model.forward_prob(prob_stats), dim=-1)[:, 1]
            pc = torch.softmax(model.forward_code(code_pooled), dim=-1)[:, 1]
            all_probs.extend((alpha * pp + (1 - alpha) * pc).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    tp = fp = tn = fn = 0
    results = []
    for idx, (prob, label) in enumerate(zip(all_probs, all_labels)):
        cve = records[idx].get("CVE-name", records[idx].get("cve_name", "unknown"))
        pred = int(prob >= 0.5)
        results.append({"CVE-name": cve, "pre-label": int(label), "predicted-label": pred, "prob_class_1": float(prob)})
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 0: tn += 1
        else: fn += 1

    acc = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"[Inference] TP:{tp} FP:{fp} TN:{tn} FN:{fn}, ACC:{acc:.4f}, prec:{prec:.4f}, rec:{rec:.4f}, f1:{f1:.4f}")

    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Results saved to: {output_path}")
    return results, {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate dual-branch membership classifier")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL from inference step")
    parser.add_argument("--embedding-model", type=str, required=True, help="Path to UnixCoder/RoBERTa model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--code-hidden-dims", type=str, default="64,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--train-percentage", type=float, default=0.2)
    parser.add_argument("--filter", type=str, default="filter_empty_pre",
                        choices=["filter_none", "filter_empty_post", "filter_empty_pre", "filter_empty_all"])
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "asl"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-output", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--use-code", type=str, default="criteria+relative",
                        choices=["all", "criteria", "relative", "criteria+relative"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    args = parser.parse_args()

    code_hidden_dims = [int(x) for x in args.code_hidden_dims.split(",")]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Dual-Branch Membership Classifier (Rich Stats + Pooled Embeddings)")
    print("=" * 80)
    print(f"device: {device}, input: {args.input}, alpha: {args.alpha}")

    embedder = CodeBERTEmbedder(model_name=args.embedding_model, device=device)

    dataset = ClassifierDataLoader(
        args.input, filter_mode=args.filter, embedder=embedder,
        limit=args.limit, batch_size=args.embedding_batch_size,
        use_cache=True, use_code=args.use_code,
    )

    all_ps, all_cp, all_labels, all_records = dataset.prob_stats, dataset.code_pooled, dataset.labels, dataset.records
    code_dim = dataset.code_dim

    c0 = [i for i, l in enumerate(all_labels) if l == 0]
    c1 = [i for i, l in enumerate(all_labels) if l == 1]
    rng = random.Random(args.seed)
    rng.shuffle(c0)
    rng.shuffle(c1)

    tc0 = int(len(c0) * args.train_percentage)
    tc1 = int(len(c1) * args.train_percentage)
    train_idx = c0[:tc0] + c1[:tc1]
    test_idx = c0[tc0:] + c1[tc1:]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    print(f"Split: label=0 train={tc0} test={len(c0)-tc0}, label=1 train={tc1} test={len(c1)-tc1}")

    train_ps = [all_ps[i] for i in train_idx]
    train_cp = [all_cp[i] for i in train_idx]
    train_labels = torch.tensor([all_labels[i] for i in train_idx])
    test_ps = [all_ps[i] for i in test_idx]
    test_cp = [all_cp[i] for i in test_idx]
    test_labels = torch.tensor([all_labels[i] for i in test_idx])
    test_records = [all_records[i] for i in test_idx]
    train_ps, test_ps = standardize_feature_lists(train_ps, test_ps)

    train_ds = ClassifierDataset(train_ps, train_cp, train_labels)
    test_ds = ClassifierDataset(test_ps, test_cp, test_labels)

    cw = [1.0 / tc0, 1.0 / tc1]
    sw = [cw[int(l)] for l in train_labels]
    sampler = WeightedRandomSampler(sw, len(sw), generator=_build_generator(args.seed + 1))

    cpu_count = os.cpu_count() or 1
    if args.num_workers is not None:
        nw = max(0, args.num_workers)
    else:
        nw = min(8, max(4, cpu_count // 16)) if cpu_count > 8 else max(0, cpu_count - 1)
    tnw = min(2, nw)
    pin = device.type == "cuda"
    loader_kw = {"collate_fn": classifier_collate_fn, "pin_memory": pin, "worker_init_fn": _seed_worker}

    train_kw = {**loader_kw, "num_workers": nw, "generator": _build_generator(args.seed + 2)}
    test_kw = {**loader_kw, "num_workers": tnw, "generator": _build_generator(args.seed + 3)}
    if nw > 0:
        train_kw.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    if tnw > 0:
        test_kw.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **train_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **test_kw)

    model = DualBranchModel(code_input_dim=code_dim, code_hidden_dims=code_hidden_dims, dropout=args.dropout).to(device)

    model, best = train_model(model, train_loader, test_loader, epochs=args.epochs,
                              lr=args.lr, device=device, loss_type=args.loss, alpha=args.alpha, save_path=args.save_path)

    print("\n" + "=" * 80)
    _, inf_metrics = inference_and_save(model=model, output_path=args.inference_output,
                                        test_loader=test_loader, records=test_records, device=device, alpha=args.alpha)
    if not best:
        best = inf_metrics
    print(f"\nBest F1: {best['f1']:.4f} (TP:{best['tp']} FP:{best['fp']} TN:{best['tn']} FN:{best['fn']})")


if __name__ == "__main__":
    main()
