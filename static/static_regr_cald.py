"""
Static Regression for Inverse Calderón Problem

This script:
1. Loads pre-trained input and output INRs
2. Extracts modulation codes (z_in, z_out) for all samples
3. Trains a mapping network (ResNet/MLP) to predict z_out from z_in
4. Evaluates the full pipeline: input → encode → map → decode → output

The final model can predict conductivity from D/N measurements on new samples.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from coral.utils.models.load_inr import create_inr_instance
from coral.mlp import ResNet

# Import the Calderón dataset
from coral.utils.data.calderon_data import CalderonCoralDataset

# Import the custom outer_step from static_inr_calderon
from static_inr_cald import graph_outer_step_calderon

def set_seed(seed=33):
    import random
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def extract_all_codes(
    inr_in,
    inr_out,
    dataloader,
    latent_dim,
    inner_steps,
    alpha_in,
    alpha_out,
    device="cuda",
):
    """
    Extract modulation codes for all samples in a dataloader.
    
    Returns:
        z_in: (N, latent_dim) input codes
        z_out: (N, latent_dim) output codes
        indices: (N,) sample indices
    """
    inr_in.eval()
    inr_out.eval()
    
    all_z_in = []
    all_z_out = []
    all_indices = []
    
    with torch.no_grad():
        for graph, idx in tqdm(dataloader, desc="Extracting codes"):
            n_samples = graph.num_graphs
            graph = graph.to(device)
            
            # Extract input codes
            input_coords = graph.input_pos
            input_features = graph.input_values
            # Handle case where follow_batch might not be set
            input_batch = graph.input_pos_batch if hasattr(graph, 'input_pos_batch') else graph.batch
            
            outputs_in = graph_outer_step_calderon(
                inr_in,
                input_coords,
                input_features,
                input_batch,
                n_samples,
                latent_dim,
                inner_steps,
                alpha_in,
                is_train=False,
            )
            z_in = outputs_in["modulations"].cpu()
            
            # Extract output codes
            output_coords = graph.output_pos
            output_features = graph.output_values
            # Handle case where follow_batch might not be set
            output_batch = graph.output_pos_batch if hasattr(graph, 'output_pos_batch') else graph.batch
            
            outputs_out = graph_outer_step_calderon(
                inr_out,
                output_coords,
                output_features,
                output_batch,
                n_samples,
                latent_dim,
                inner_steps,
                alpha_out,
                is_train=False,
            )
            z_out = outputs_out["modulations"].cpu()
            
            all_z_in.append(z_in)
            all_z_out.append(z_out)
            all_indices.extend(idx.tolist() if torch.is_tensor(idx) else list(idx))
    
    z_in = torch.cat(all_z_in, dim=0)
    z_out = torch.cat(all_z_out, dim=0)
    
    return z_in, z_out, all_indices


def evaluate_full_pipeline(
    inr_in,
    inr_out,
    mapper,
    dataloader,
    latent_dim,
    inner_steps,
    alpha_in,
    mu_in,
    std_in,
    mu_out,
    std_out,
    device="cuda",
    data_stats=None,
):
    """
    Evaluate the full pipeline: input → encode → map → decode → output.
    
    Returns comprehensive metrics in both normalized and denormalized space.
    """
    inr_in.eval()
    inr_out.eval()
    mapper.eval()
    
    # Storage for per-sample errors
    l1_errors_normalized = []
    l2_errors_normalized = []
    l1_errors_denormalized = []
    l2_errors_denormalized = []
    
    # For aggregate metrics
    total_mse = 0.0
    total_rel_error = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for graph, idx in tqdm(dataloader, desc="Evaluating"):
            batch_size = graph.num_graphs
            graph = graph.to(device)
            
            # 1. Encode input D/N data
            input_coords = graph.input_pos
            input_features = graph.input_values
            # Handle case where follow_batch might not be set
            input_batch = graph.input_pos_batch if hasattr(graph, 'input_pos_batch') else graph.batch
            
            outputs_in = graph_outer_step_calderon(
                inr_in,
                input_coords,
                input_features,
                input_batch,
                batch_size,
                latent_dim,
                inner_steps,
                alpha_in,
                is_train=False,
            )
            z_in = outputs_in["modulations"]
            
            # 2. Normalize input codes
            z_in_norm = (z_in - mu_in.to(device)) / std_in.to(device)
            
            # 3. Map to output codes
            z_out_pred = mapper(z_in_norm)
            
            # 4. Denormalize output codes
            z_out_pred = z_out_pred * std_out.to(device) + mu_out.to(device)
            
            # 5. Decode to conductivity
            output_coords = graph.output_pos
            # Handle case where follow_batch might not be set
            output_batch = graph.output_pos_batch if hasattr(graph, 'output_pos_batch') else graph.batch
            
            cond_pred = inr_out.modulated_forward(output_coords, z_out_pred[output_batch])
            
            # 6. Compute errors per sample
            cond_true = graph.output_values
            
            # Get the pointer to separate samples in the batch
            # output_pos_batch tells us which sample each node belongs to
            for i in range(batch_size):
                mask = (output_batch == i)
                pred_i = cond_pred[mask]
                true_i = cond_true[mask]
                
                # L1 relative error (normalized space)
                l1_rel = (torch.abs(pred_i - true_i).sum() / 
                         (torch.abs(true_i).sum() + 1e-8))
                l1_errors_normalized.append(l1_rel.cpu().item())
                
                # L2 relative error (normalized space)
                l2_rel = (torch.norm(pred_i - true_i, p=2) / 
                         (torch.norm(true_i, p=2) + 1e-8))
                l2_errors_normalized.append(l2_rel.cpu().item())
                
                # Denormalized errors if stats available
                if data_stats is not None and 'conductivity_mean' in data_stats:
                    cond_mean = data_stats['conductivity_mean']
                    cond_std = data_stats['conductivity_std']
                    
                    pred_denorm = pred_i.cpu() * cond_std + cond_mean
                    true_denorm = true_i.cpu() * cond_std + cond_mean
                    
                    l1_rel_denorm = (torch.abs(pred_denorm - true_denorm).sum() / 
                                    (torch.abs(true_denorm).sum() + 1e-8))
                    l1_errors_denormalized.append(l1_rel_denorm.item())
                    
                    l2_rel_denorm = (torch.norm(pred_denorm - true_denorm, p=2) / 
                                    (torch.norm(true_denorm, p=2) + 1e-8))
                    l2_errors_denormalized.append(l2_rel_denorm.item())
            
            # Aggregate metrics
            mse = ((cond_pred - cond_true) ** 2).mean()
            rel_error = ((cond_pred - cond_true) ** 2).sum() / ((cond_true) ** 2).sum() 
            
            total_mse += mse.item() * batch_size
            total_rel_error += rel_error.item() * batch_size
            n_samples += batch_size
    
    # Compile comprehensive results
    results = {
        'aggregate': {
            'mse': total_mse / n_samples,
            'rel_error': total_rel_error / n_samples,
        },
        'normalized': {
            'l1': {
                'mean': np.mean(l1_errors_normalized),
                'median': np.median(l1_errors_normalized),
                'std': np.std(l1_errors_normalized),
                'min': np.min(l1_errors_normalized),
                'max': np.max(l1_errors_normalized),
                'percentile_25': np.percentile(l1_errors_normalized, 25),
                'percentile_75': np.percentile(l1_errors_normalized, 75),
            },
            'l2': {
                'mean': np.mean(l2_errors_normalized),
                'median': np.median(l2_errors_normalized),
                'std': np.std(l2_errors_normalized),
                'min': np.min(l2_errors_normalized),
                'max': np.max(l2_errors_normalized),
                'percentile_25': np.percentile(l2_errors_normalized, 25),
                'percentile_75': np.percentile(l2_errors_normalized, 75),
            }
        }
    }
    
    if len(l1_errors_denormalized) > 0:
        results['denormalized'] = {
            'l1': {
                'mean': np.mean(l1_errors_denormalized),
                'median': np.median(l1_errors_denormalized),
                'std': np.std(l1_errors_denormalized),
                'min': np.min(l1_errors_denormalized),
                'max': np.max(l1_errors_denormalized),
                'percentile_25': np.percentile(l1_errors_denormalized, 25),
                'percentile_75': np.percentile(l1_errors_denormalized, 75),
            },
            'l2': {
                'mean': np.mean(l2_errors_denormalized),
                'median': np.median(l2_errors_denormalized),
                'std': np.std(l2_errors_denormalized),
                'min': np.min(l2_errors_denormalized),
                'max': np.max(l2_errors_denormalized),
                'percentile_25': np.percentile(l2_errors_denormalized, 25),
                'percentile_75': np.percentile(l2_errors_denormalized, 75),
            }
        }
    
    return results


def print_metrics_table(results, title="Test Metrics"):
    """Pretty print metrics in a table format"""
    print("\n" + "="*90)
    print(f"{title:^90}")
    print("="*90)
    
    if 'aggregate' in results:
        print(f"\n📈 AGGREGATE METRICS:")
        print(f"  MSE: {results['aggregate']['mse']:.6f}")
        print(f"  Relative Error: {results['aggregate']['rel_error']:.6f}")

    if 'normalized' in results:
        print("\n📊 NORMALIZED SPACE:")
        print("-" * 90)
        print(f"{'Metric':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Q1':<10} {'Q3':<10} {'Max':<10}")
        print("-" * 90)

        l1 = results['normalized']['l1']
        print(f"{'L1 Rel Error':<15} {l1['mean']:<10.6f} {l1['median']:<10.6f} "
              f"{l1['std']:<10.6f} {l1['min']:<10.6f} {l1['percentile_25']:<10.6f} "
              f"{l1['percentile_75']:<10.6f} {l1['max']:<10.6f}")

        l2 = results['normalized']['l2']
        print(f"{'L2 Rel Error':<15} {l2['mean']:<10.6f} {l2['median']:<10.6f} "
              f"{l2['std']:<10.6f} {l2['min']:<10.6f} {l2['percentile_25']:<10.6f} "
              f"{l2['percentile_75']:<10.6f} {l2['max']:<10.6f}")

    if 'denormalized' in results:
        print("\n📊 DENORMALIZED SPACE:")
        print("-" * 90)
        print(f"{'Metric':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Q1':<10} {'Q3':<10} {'Max':<10}")
        print("-" * 90)

        l1 = results['denormalized']['l1']
        print(f"{'L1 Rel Error':<15} {l1['mean']:<10.6f} {l1['median']:<10.6f} "
              f"{l1['std']:<10.6f} {l1['min']:<10.6f} {l1['percentile_25']:<10.6f} "
              f"{l1['percentile_75']:<10.6f} {l1['max']:<10.6f}")

        l2 = results['denormalized']['l2']
        print(f"{'L2 Rel Error':<15} {l2['mean']:<10.6f} {l2['median']:<10.6f} "
              f"{l2['std']:<10.6f} {l2['min']:<10.6f} {l2['percentile_25']:<10.6f} "
              f"{l2['percentile_75']:<10.6f} {l2['max']:<10.6f}")

    print("="*90 + "\n")


def save_metrics_to_file(results, filepath='coral_metrics.txt'):
    """Save metrics to a text file"""
    with open(filepath, 'w') as f:
        f.write("="*90 + "\n")
        f.write("CORAL FINAL TEST METRICS\n")
        f.write("="*90 + "\n\n")
        
        if 'aggregate' in results:
            f.write("AGGREGATE METRICS:\n")
            f.write(f"  MSE: {results['aggregate']['mse']:.8f}\n")
            f.write(f"  Relative Error: {results['aggregate']['rel_error']:.8f}\n\n")

        if 'normalized' in results:
            f.write("NORMALIZED SPACE:\n")
            f.write("-" * 90 + "\n")

            l1 = results['normalized']['l1']
            f.write(f"L1 Relative Error:\n")
            f.write(f"  Mean:   {l1['mean']:.8f}\n")
            f.write(f"  Median: {l1['median']:.8f}\n")
            f.write(f"  Std:    {l1['std']:.8f}\n")
            f.write(f"  Min:    {l1['min']:.8f}\n")
            f.write(f"  Q1:     {l1['percentile_25']:.8f}\n")
            f.write(f"  Q3:     {l1['percentile_75']:.8f}\n")
            f.write(f"  Max:    {l1['max']:.8f}\n\n")

            l2 = results['normalized']['l2']
            f.write(f"L2 Relative Error:\n")
            f.write(f"  Mean:   {l2['mean']:.8f}\n")
            f.write(f"  Median: {l2['median']:.8f}\n")
            f.write(f"  Std:    {l2['std']:.8f}\n")
            f.write(f"  Min:    {l2['min']:.8f}\n")
            f.write(f"  Q1:     {l2['percentile_25']:.8f}\n")
            f.write(f"  Q3:     {l2['percentile_75']:.8f}\n")
            f.write(f"  Max:    {l2['max']:.8f}\n\n")

        if 'denormalized' in results:
            f.write("DENORMALIZED SPACE:\n")
            f.write("-" * 90 + "\n")

            l1 = results['denormalized']['l1']
            f.write(f"L1 Relative Error:\n")
            f.write(f"  Mean:   {l1['mean']:.8f}\n")
            f.write(f"  Median: {l1['median']:.8f}\n")
            f.write(f"  Std:    {l1['std']:.8f}\n")
            f.write(f"  Min:    {l1['min']:.8f}\n")
            f.write(f"  Q1:     {l1['percentile_25']:.8f}\n")
            f.write(f"  Q3:     {l1['percentile_75']:.8f}\n")
            f.write(f"  Max:    {l1['max']:.8f}\n\n")

            l2 = results['denormalized']['l2']
            f.write(f"L2 Relative Error:\n")
            f.write(f"  Mean:   {l2['mean']:.8f}\n")
            f.write(f"  Median: {l2['median']:.8f}\n")
            f.write(f"  Std:    {l2['std']:.8f}\n")
            f.write(f"  Min:    {l2['min']:.8f}\n")
            f.write(f"  Q1:     {l2['percentile_25']:.8f}\n")
            f.write(f"  Q3:     {l2['percentile_75']:.8f}\n")
            f.write(f"  Max:    {l2['max']:.8f}\n")

        f.write("="*90 + "\n")

    print(f"✓ Metrics saved to {filepath}")


@hydra.main(config_path="config/calderon/", config_name="regression.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function for the mapping network."""
    
    torch.set_default_dtype(torch.float32)
    
    # === Extract config values ===
    # Data
    data_path = cfg.data.data_path
    seed = cfg.data.seed
    
    # Optimization
    batch_size = cfg.optim.batch_size
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    epochs = cfg.optim.epochs
    
    # INR (pre-trained)
    inr_run_name = cfg.inr.run_name
    inner_steps = cfg.inr.inner_steps
    
    # Model
    model_width = cfg.model.width
    model_depth = cfg.model.depth
    model_dropout = cfg.model.dropout
    model_activation = cfg.model.activation
    
    # Wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_name = cfg.wandb.name
    run_id = cfg.wandb.id
    
    # === Setup wandb ===
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    run_name = wandb.run.name
    
    print(f"Run ID: {run.id}")
    print(f"Run name: {run_name}")
    
    # Results directory
    RESULTS_DIR = Path(os.getenv("WANDB_DIR", ".")) / "calderon" / "regression"
    INR_DIR = Path(os.getenv("WANDB_DIR", ".")) / "calderon" / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    
    # === Load pre-trained INRs ===
    print(f"Loading INRs from: {INR_DIR / f'{inr_run_name}.pt'}")
    inr_checkpoint = torch.load(INR_DIR / f"{inr_run_name}.pt", weights_only=False)
    
    inr_cfg = inr_checkpoint["cfg"]
    latent_dim_in = inr_cfg.inr_in.latent_dim
    latent_dim_out = inr_cfg.inr_out.latent_dim
    data_stats = inr_checkpoint.get("stats", None)
    
    # Use the seed from the INR training for consistency
    seed = inr_cfg.data.seed
    set_seed(seed)
    
    # Get data dimensions
    ntrain = inr_cfg.data.ntrain
    ntest = inr_cfg.data.ntest
    
    # Create INR instances
    input_dim = 2
    output_dim_in = 40  # D/N values
    output_dim_out = 1  # conductivity
    
    inr_cfg.inr = inr_cfg.inr_in
    inr_in = create_inr_instance(inr_cfg, input_dim=input_dim, output_dim=output_dim_in, device="cuda")
    inr_in.load_state_dict(inr_checkpoint["inr_in"])
    inr_in.eval()
    
    inr_cfg.inr = inr_cfg.inr_out
    inr_out = create_inr_instance(inr_cfg, input_dim=input_dim, output_dim=output_dim_out, device="cuda")
    inr_out.load_state_dict(inr_checkpoint["inr_out"])
    inr_out.eval()
    
    alpha_in = inr_checkpoint["alpha_in"]
    alpha_out = inr_checkpoint["alpha_out"]
    
    print(f"Loaded INRs with latent_dim={latent_dim_in}")
    print(f"alpha_in={alpha_in.item():.4f}, alpha_out={alpha_out.item():.4f}")
    
    # === Create datasets ===
    trainset = CalderonCoralDataset(
        data_path=data_path,
        split="train",
        latent_dim=latent_dim_in,
        normalize=True,
        stats=data_stats,
        ntrain=ntrain,
        ntest=ntest,
        seed=seed,
    )
    
    testset = CalderonCoralDataset(
        data_path=data_path,
        split="test",
        latent_dim=latent_dim_in,
        normalize=True,
        stats=trainset.stats,
        ntrain=ntrain,
        ntest=ntest,
        seed=seed,
    )
    
    train_loader = DataLoader(
        dataset=trainset, 
        batch_size=batch_size, 
        shuffle=False,
        follow_batch=['input_pos', 'output_pos'],
    )
    test_loader = DataLoader(
        dataset=testset, 
        batch_size=batch_size, 
        shuffle=False,
        follow_batch=['input_pos', 'output_pos'],
    )
    
    print(f"Train: {len(trainset)}, Test: {len(testset)}")
    
    # Add tags
    run.tags = ("calderon-regression", f"inr={inr_run_name}")
    
    # === Extract all codes ===
    print("\n" + "="*50)
    print("Extracting modulation codes...")
    print("="*50)
    
    z_in_train, z_out_train, _ = extract_all_codes(
        inr_in, inr_out, train_loader, latent_dim_in, inner_steps, alpha_in, alpha_out
    )
    
    z_in_test, z_out_test, _ = extract_all_codes(
        inr_in, inr_out, test_loader, latent_dim_in, inner_steps, alpha_in, alpha_out
    )
    
    print(f"Train codes: z_in={z_in_train.shape}, z_out={z_out_train.shape}")
    print(f"Test codes: z_in={z_in_test.shape}, z_out={z_out_test.shape}")
    
    # === Normalize codes ===
    mu_in = z_in_train.mean(dim=0)
    std_in = z_in_train.std(dim=0) + 1e-8
    mu_out = z_out_train.mean(dim=0)
    std_out = z_out_train.std(dim=0) + 1e-8
    
    z_in_train_norm = (z_in_train - mu_in) / std_in
    z_in_test_norm = (z_in_test - mu_in) / std_in
    
    z_out_train_norm = (z_out_train - mu_out) / std_out
    z_out_test_norm = (z_out_test - mu_out) / std_out
    
    print(f"\nCode statistics:")
    print(f"  z_in: mean={z_in_train.mean():.4f}, std={z_in_train.std():.4f}")
    print(f"  z_out: mean={z_out_train.mean():.4f}, std={z_out_train.std():.4f}")
    
    # === Create mapping network ===
    mapper = ResNet(
        input_dim=latent_dim_in,
        hidden_dim=model_width,
        output_dim=latent_dim_out,
        depth=model_depth,
        dropout=model_dropout,
        activation=model_activation,
    ).cuda()
    
    print(f"\nMapper parameters: {sum(p.numel() for p in mapper.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # === Training loop for mapper ===
    print("\n" + "="*50)
    print("Training mapping network...")
    print("="*50)
    
    best_test_loss = np.inf
    
    for epoch in range(epochs):
        mapper.train()
        
        # Shuffle training data
        perm = torch.randperm(len(z_in_train_norm))
        z_in_shuffled = z_in_train_norm[perm]
        z_out_shuffled = z_out_train_norm[perm]
        
        train_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(z_in_shuffled), batch_size):
            z_in_batch = z_in_shuffled[i:i+batch_size].cuda()
            z_out_batch = z_out_shuffled[i:i+batch_size].cuda()
            
            z_pred = mapper(z_in_batch)
            loss = ((z_pred - z_out_batch) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        scheduler.step()
        
        # Test evaluation
        mapper.eval()
        with torch.no_grad():
            z_pred_test = mapper(z_in_test_norm.cuda())
            test_loss = ((z_pred_test - z_out_test_norm.cuda()) ** 2).mean().item()
        
        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train={train_loss:.6f}, Test={test_loss:.6f}")
        
        wandb.log(
            {
                "code_train_loss": train_loss,
                "code_test_loss": test_loss,
                "lr": scheduler.get_last_lr()[0],
            },
            step=epoch,
        )
        
        # Save best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            
            torch.save(
                {
                    "cfg": cfg,
                    "inr_cfg": inr_cfg,
                    "epoch": epoch,
                    "mapper": mapper.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "test_loss": test_loss,
                    "mu_in": mu_in,
                    "std_in": std_in,
                    "mu_out": mu_out,
                    "std_out": std_out,
                    "alpha_in": alpha_in,
                    "alpha_out": alpha_out,
                    "inr_run_name": inr_run_name,
                    "data_stats": trainset.stats,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )
    
    print(f"\nBest test loss: {best_test_loss:.6f}")
    
    # === Full pipeline evaluation ===
    print("\n" + "="*50)
    print("Full pipeline evaluation...")
    print("="*50)
    
    # Reload best model
    best_checkpoint = torch.load(f"{RESULTS_DIR}/{run_name}.pt", weights_only=False)
    mapper.load_state_dict(best_checkpoint["mapper"])
    
    # Evaluate on test set with follow_batch
    test_loader = DataLoader(
        dataset=testset, 
        batch_size=batch_size, 
        shuffle=False,
        follow_batch=['input_pos', 'output_pos'],
    )
    
    test_metrics = evaluate_full_pipeline(
        inr_in,
        inr_out,
        mapper,
        test_loader,
        latent_dim_in,
        inner_steps,
        alpha_in,
        mu_in,
        std_in,
        mu_out,
        std_out,
        data_stats=trainset.stats,  # Pass stats for denormalization
    )
    
    # Print comprehensive metrics
    print_metrics_table(test_metrics, title="CORAL Test Metrics")
    
    # Save metrics to file
    save_metrics_to_file(test_metrics, filepath=f"{RESULTS_DIR}/{run_name}_metrics.txt")
    
    # Log to wandb
    wandb.log(
        {
            "final_test_mse": test_metrics["aggregate"]["mse"],
            "final_test_rel_error": test_metrics["aggregate"]["rel_error"],
            "final_l1_normalized_mean": test_metrics["normalized"]["l1"]["mean"],
            "final_l2_normalized_mean": test_metrics["normalized"]["l2"]["mean"],
        },
    )
    
    if "denormalized" in test_metrics:
        wandb.log(
            {
                "final_l1_denormalized_mean": test_metrics["denormalized"]["l1"]["mean"],
                "final_l2_denormalized_mean": test_metrics["denormalized"]["l2"]["mean"],
            },
        )
    
    wandb.finish()
    return test_metrics


if __name__ == "__main__":
    main()