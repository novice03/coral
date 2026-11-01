#!/usr/bin/env python3
"""
Save CORAL predictions to .npz file for MCB Poisson datasets.

CORAL is a two-stage model:
1. INR stage: Encodes input field (boundary BCs) into modulation code z_in
2. Regression stage: Maps z_in → z_out, then decodes to output field (solution)

Usage:
    python save_mcb_preds.py \
        --dataset gear \
        --inr_run_name "inr_gear" \
        --regr_run_name "regr_gear" \
        --output_dir ~/model_predictions \
        --sample_indices 10 50 100
"""

import torch
import argparse
import numpy as np
import random
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append('/fs/nexus-scratch/pulijala/coral')
sys.path.append('/fs/nexus-scratch/pulijala/coral/static')

from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from coral.utils.models.load_inr import create_inr_instance
from coral.mlp import ResNet
from mcb_data import MCBCoralDataset, extract_shape_ids_from_file
from static_inr_mcb import graph_outer_step_mcb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_mcb_models(inr_checkpoint_path, regr_checkpoint_path, device, config_dir):
    """
    Load CORAL INR and regression models for MCB.

    Architecture from YAML configs, weights/stats from checkpoints.
    """
    inr_yaml  = OmegaConf.load(os.path.join(config_dir, 'static_inr.yaml'))
    regr_yaml = OmegaConf.load(os.path.join(config_dir, 'regression.yaml'))

    # ---- Load INR checkpoint ----
    print(f"Loading INR checkpoint from: {inr_checkpoint_path}")
    inr_checkpoint = torch.load(inr_checkpoint_path, map_location=device, weights_only=False)

    data_stats = inr_checkpoint.get("stats", None)
    latent_dim = inr_yaml.inr_in.latent_dim

    inr_yaml.inr = inr_yaml.inr_in
    inr_in = create_inr_instance(inr_yaml, input_dim=3, output_dim=1, device=device)
    inr_in.load_state_dict(inr_checkpoint["inr_in"])
    inr_in.eval()

    inr_yaml.inr = inr_yaml.inr_out
    inr_out = create_inr_instance(inr_yaml, input_dim=3, output_dim=1, device=device)
    inr_out.load_state_dict(inr_checkpoint["inr_out"])
    inr_out.eval()

    alpha_in  = inr_checkpoint["alpha_in"]
    alpha_out = inr_checkpoint["alpha_out"]

    # Read data paths from saved cfg inside INR checkpoint
    inr_cfg = inr_checkpoint.get("cfg", None)

    print(f"  Latent dim: {latent_dim}")
    print(f"  alpha_in: {alpha_in.item():.4f}, alpha_out: {alpha_out.item():.4f}")

    # ---- Load regression checkpoint ----
    print(f"\nLoading regression checkpoint from: {regr_checkpoint_path}")
    regr_checkpoint = torch.load(regr_checkpoint_path, map_location=device, weights_only=False)

    mapper = ResNet(
        input_dim=latent_dim,
        hidden_dim=regr_yaml.model.width,
        output_dim=latent_dim,
        depth=regr_yaml.model.depth,
        dropout=regr_yaml.model.dropout,
        activation=regr_yaml.model.activation,
    ).to(device)
    mapper.load_state_dict(regr_checkpoint["mapper"])
    mapper.eval()

    mu_in  = regr_checkpoint["mu_in"]
    std_in = regr_checkpoint["std_in"]
    mu_out = regr_checkpoint["mu_out"]
    std_out = regr_checkpoint["std_out"]

    inner_steps = regr_yaml.inr.inner_steps

    # Prefer data paths from regression cfg (dataset-specific)
    regr_cfg = regr_checkpoint.get("cfg", None)
    if regr_cfg is not None:
        data_root       = regr_cfg.data.data_root
        train_split_file = regr_cfg.data.train_split_file
        test_split_file  = regr_cfg.data.test_split_file
    elif inr_cfg is not None:
        data_root       = inr_cfg.data.data_root
        train_split_file = inr_cfg.data.train_split_file
        test_split_file  = inr_cfg.data.test_split_file
    else:
        data_root = train_split_file = test_split_file = None

    seed = 42

    print(f"  inner_steps (regression.yaml): {inner_steps}")
    print(f"  data_root: {data_root}")
    print(f"  test_split_file: {test_split_file}")
    print(f"  Mapper params: {sum(p.numel() for p in mapper.parameters()):,}")

    # data_stats from regression checkpoint takes precedence (computed from train set)
    if regr_checkpoint.get("data_stats") is not None:
        data_stats = regr_checkpoint["data_stats"]

    return {
        'inr_in':          inr_in,
        'inr_out':         inr_out,
        'mapper':          mapper,
        'alpha_in':        alpha_in,
        'alpha_out':       alpha_out,
        'mu_in':           mu_in,
        'std_in':          std_in,
        'mu_out':          mu_out,
        'std_out':         std_out,
        'latent_dim':      latent_dim,
        'data_stats':      data_stats,
        'inner_steps':     inner_steps,
        'data_root':       data_root,
        'train_split_file': train_split_file,
        'test_split_file': test_split_file,
        'seed':            seed,
    }


def run_mcb_inference(models, graph, inner_steps, device):
    """
    Run CORAL inference: boundary BCs → encode → map → decode → solution
    """
    inr_in    = models['inr_in']
    inr_out   = models['inr_out']
    mapper    = models['mapper']
    alpha_in  = models['alpha_in']
    mu_in     = models['mu_in'].to(device)
    std_in    = models['std_in'].to(device)
    mu_out    = models['mu_out'].to(device)
    std_out   = models['std_out'].to(device)
    latent_dim = models['latent_dim']

    batch_size = graph.num_graphs

    # 1. Encode input (boundary BCs) to get z_in
    input_coords  = graph.input_pos
    input_features = graph.input_values
    input_batch   = graph.input_pos_batch

    outputs_in = graph_outer_step_mcb(
        inr_in,
        input_coords,
        input_features,
        input_batch,
        batch_size,
        latent_dim,
        inner_steps,
        alpha_in,
        is_train=False,
        device=str(input_coords.device),
    )
    z_in = outputs_in["modulations"]

    # 2. Normalize, map, denormalize
    z_in_norm  = (z_in - mu_in) / std_in
    z_out_pred = mapper(z_in_norm)
    z_out_pred = z_out_pred * std_out + mu_out

    # 3. Decode to solution field
    output_coords = graph.output_pos
    output_batch  = graph.output_pos_batch

    u_pred = inr_out.modulated_forward(output_coords, z_out_pred[output_batch])

    return u_pred


def main():
    parser = argparse.ArgumentParser(description='Save CORAL MCB predictions to .npz file')

    parser.add_argument('--dataset', type=str, default='gear',
                        choices=['gear', 'multishape', 'motor', 'nut', 'screwsbolts'],
                        help='MCB dataset name')
    parser.add_argument('--inr_run_name', type=str, default=None,
                        help='INR checkpoint name (default: inr_<dataset>)')
    parser.add_argument('--regr_run_name', type=str, default=None,
                        help='Regression checkpoint name (default: regr_<dataset>)')
    parser.add_argument('--coral_dir', type=str,
                        default='/fs/nexus-scratch/pulijala/coral/static',
                        help='Base directory for CORAL checkpoints and configs')

    # Data paths (override checkpoint-embedded paths if needed)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root directory')
    parser.add_argument('--test_split_file', type=str, default=None,
                        help='Override test split file path')
    parser.add_argument('--train_split_file', type=str, default=None,
                        help='Override train split file (used to compute stats if needed)')

    # Output
    parser.add_argument('--output_dir', type=str,
                        default=os.path.expanduser('~/model_predictions'),
                        help='Directory to save predictions')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output filename (default: coral_mcb_<dataset>_predictions.npz)')

    # Sample selection
    parser.add_argument('--sample_indices', type=int, nargs='+',
                        help='Specific sample indices to run inference on')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of random samples (ignored if --sample_indices provided)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: 42)')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Derive default run names from dataset
    inr_run_name  = args.inr_run_name  or f'inr_{args.dataset}'
    regr_run_name = args.regr_run_name or f'regr_{args.dataset}'
    output_name   = args.output_name   or f'coral_mcb_{args.dataset}_predictions.npz'

    config_dir = os.path.join(args.coral_dir, 'config', 'mcb')
    inr_checkpoint_path  = os.path.join(args.coral_dir, 'mcb', 'inr',        f'{inr_run_name}.pt')
    regr_checkpoint_path = os.path.join(args.coral_dir, 'mcb', 'regression', f'{regr_run_name}.pt')

    print("\n" + "=" * 60)
    print("Loading CORAL MCB models...")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = load_mcb_models(inr_checkpoint_path, regr_checkpoint_path, device, config_dir)

    inner_steps = models['inner_steps']
    seed        = args.seed if args.seed is not None else models['seed']
    data_stats  = models['data_stats']

    # Allow CLI overrides for data paths
    data_root       = args.data_root       or models['data_root']
    test_split_file  = args.test_split_file  or models['test_split_file']
    train_split_file = args.train_split_file or models['train_split_file']

    print(f"Using device: {device}")
    set_seed(seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_name)

    # ---- Load test dataset ----
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    test_shapes = extract_shape_ids_from_file(test_split_file)
    existing_shapes = set(os.listdir(data_root))
    test_shapes = [s for s in test_shapes if s in existing_shapes]
    print(f"Test shapes: {len(test_shapes)}")

    testset = MCBCoralDataset(
        data_root=data_root,
        shape_ids=test_shapes,
        split="test",
        latent_dim=models['latent_dim'],
        normalize=True,
        stats=data_stats,
    )

    # If stats are missing, compute from training set
    if data_stats is None:
        train_shapes = extract_shape_ids_from_file(train_split_file)
        train_shapes = [s for s in train_shapes if s in existing_shapes]
        trainset = MCBCoralDataset(
            data_root=data_root,
            shape_ids=train_shapes,
            split="train",
            latent_dim=models['latent_dim'],
            normalize=True,
        )
        data_stats = trainset.stats

    test_size = len(testset)
    print(f"Test dataset size: {test_size}")

    # Determine sample indices
    if args.sample_indices:
        sample_indices = args.sample_indices
    elif args.num_samples:
        sample_indices = random.sample(range(test_size), min(args.num_samples, test_size))
    else:
        sample_indices = list(range(test_size))

    sample_indices = [idx for idx in sample_indices if idx < test_size]
    print(f"\nRunning on {len(sample_indices)} samples with inner_steps={inner_steps}")

    test_loader = DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False,
        follow_batch=['input_pos', 'output_pos'],
    )

    # Pre-load all graphs indexed by local index
    all_graphs = {}
    for (graph, idx) in test_loader:
        local_idx = int(idx.item())
        all_graphs[local_idx] = graph

    # ---- Run inference ----
    print("\n" + "=" * 60)
    print("Running CORAL inference...")
    print("=" * 60)

    save_dict = {
        'model_name':    'CORAL_MCB',
        'dataset':       args.dataset,
        'sample_indices': np.array(sample_indices),
        'inr_run_name':  inr_run_name,
        'regr_run_name': regr_run_name,
        'inner_steps':   inner_steps,
    }

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            print(f"\nProcessing sample {i+1}/{len(sample_indices)} (index {idx})...")

            if idx not in all_graphs:
                print(f"  Warning: index {idx} not in test set, skipping")
                continue

            graph = all_graphs[idx].to(device)

            u_pred = run_mcb_inference(models, graph, inner_steps, device)

            input_values  = graph.input_values.squeeze(-1)   # (M,) normalized BCs
            output_values = graph.output_values.squeeze(-1)  # (N,) normalized solution
            input_coords  = graph.input_pos                  # (M, 3)
            output_coords = graph.output_pos                 # (N, 3)

            # Denormalize
            sol_mean = data_stats['sol_mean']
            sol_std  = data_stats['sol_std']
            bd_mean  = data_stats['bd_vals_mean']
            bd_std   = data_stats['bd_vals_std']

            gt_denorm   = (output_values.cpu() * sol_std + sol_mean).numpy()
            pred_denorm = (u_pred.squeeze(-1).cpu() * sol_std + sol_mean).numpy()
            bc_denorm   = (input_values.cpu() * bd_std + bd_mean).numpy()

            rel_l2 = np.linalg.norm(gt_denorm - pred_denorm) / (np.linalg.norm(gt_denorm) + 1e-8)
            print(f"  Relative L2 Error: {rel_l2:.6f}")

            save_dict[f'bc_{i}']           = bc_denorm
            save_dict[f'input_coords_{i}'] = input_coords.cpu().numpy()
            save_dict[f'gt_{i}']           = gt_denorm
            save_dict[f'pred_{i}']         = pred_denorm
            save_dict[f'output_coords_{i}'] = output_coords.cpu().numpy()
            save_dict[f'rel_l2_{i}']       = rel_l2

    # ---- Save ----
    print(f"\n" + "=" * 60)
    print(f"Saving predictions to: {output_path}")
    print("=" * 60)
    np.savez(output_path, **save_dict)

    rel_l2_errors = [save_dict[f'rel_l2_{i}'] for i in range(len(sample_indices))
                     if f'rel_l2_{i}' in save_dict]
    print(f"\nSummary:")
    print(f"  Samples saved:      {len(rel_l2_errors)}")
    print(f"  Mean Relative L2:   {np.mean(rel_l2_errors):.6f} ± {np.std(rel_l2_errors):.6f}")
    print(f"  Median Relative L2: {np.median(rel_l2_errors):.6f}")
    print(f"  Output file:        {output_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
