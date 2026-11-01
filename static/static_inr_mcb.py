"""
Static INR Training for MCB Multi-Shape Poisson Problem

This script trains two INRs:
1. Input INR: Maps (boundary_coords ∪ interior_coords) → (boundary_values ∪ source_term)
2. Output INR: Maps mesh_nodes → solution

The INRs use MAML-style meta-learning to find modulation codes that condition
the network to fit each individual sample.

Key features:
- 3D coordinates (input_dim=3)
- Multi-shape dataset (200 train shapes, 20 test shapes)
- No explicit conditioning on bd_A, bd_B, or shape ID
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from coral.utils.models.load_inr import create_inr_instance

# Import the MCB CORAL dataset
from mcb_data import MCBCoralDataset, extract_shape_ids_from_file


def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def graph_outer_step_mcb(
    func_rep,
    coords,
    features,
    batch_index,
    n_graphs,
    latent_dim,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    use_rel_loss=False,
    device="cuda",
):
    """
    Outer step for CORAL training adapted for MCB Poisson problem.
    
    Args:
        func_rep: INR model with modulated_forward method
        coords: (total_N, 3) concatenated coordinates
        features: (total_N, 1) concatenated target values
        batch_index: (total_N,) batch assignment for each point
        n_graphs: number of graphs in batch
        latent_dim: dimension of modulation codes
        inner_steps: number of inner loop iterations
        inner_lr: inner loop learning rate (can be learnable)
        is_train: whether to compute gradients for outer loop
        return_reconstructions: whether to return predicted values
        use_rel_loss: whether to compute relative loss
        device: device to use
    
    Returns:
        dict with 'loss', 'modulations', and optionally 'reconstructions', 'rel_loss'
    """
    func_rep.zero_grad()
    
    # Initialize modulations to zero
    modulations = torch.zeros(n_graphs, latent_dim, device=device).requires_grad_()
    
    # Run inner loop to fit modulations
    for step in range(inner_steps):
        modulations = graph_inner_loop_step_mcb(
            func_rep,
            modulations,
            coords,
            features,
            batch_index,
            inner_lr,
            is_train,
        )
    
    # Compute outer loss
    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        loss = ((features_recon - features) ** 2).mean()
    
    outputs = {
        "loss": loss,
        "modulations": modulations,
    }
    
    if return_reconstructions:
        outputs["reconstructions"] = features_recon
    
    if use_rel_loss:
        rel_loss = ((features_recon - features) ** 2).sum() / ((features) ** 2).sum() 
        outputs["rel_loss"] = rel_loss
    
    return outputs


def graph_inner_loop_step_mcb(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
):
    """
    Single inner loop step for fitting modulations.
    
    Uses gradient descent on the reconstruction loss w.r.t. modulations.
    """
    batch_size = modulations.shape[0]
    
    if not modulations.requires_grad:
        modulations = modulations.requires_grad_()
    
    with torch.enable_grad():
        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        # Scale loss by batch size for proper gradient magnitude
        loss = ((features_recon - features) ** 2).mean() * batch_size
        
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train,
        )[0]
    
    new_modulations = modulations - inner_lr * grad
    
    if not is_train:
        new_modulations = new_modulations.detach().requires_grad_()
    
    return new_modulations


@hydra.main(config_path="config/mcb/", config_name="static_inr.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    torch.set_default_dtype(torch.float32)
    
    # === Extract config values ===
    data_root = cfg.data.data_root
    train_split_file = cfg.data.train_split_file
    test_split_file = cfg.data.test_split_file
    subsample_interior = cfg.data.subsample_interior
    seed = cfg.data.seed
    
    batch_size = cfg.optim.batch_size
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    
    latent_dim = cfg.inr_in.latent_dim
    
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
    RESULTS_DIR = Path(os.getenv("WANDB_DIR", ".")) / "mcb" / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)
    
    set_seed(seed)
    
    run.tags = ("mcb-inr", f"latent_dim={latent_dim}")
    
    # === Create datasets ===
    print("\n" + "="*70)
    print("Setting up datasets...")
    print("="*70)
    
    # Extract shape IDs
    train_shapes = extract_shape_ids_from_file(train_split_file)
    test_shapes = extract_shape_ids_from_file(test_split_file)
    
    # Filter to existing shapes
    existing_shapes = set(os.listdir(data_root))
    train_shapes = [s for s in train_shapes if s in existing_shapes]
    test_shapes = [s for s in test_shapes if s in existing_shapes]
    
    print(f"Train shapes: {len(train_shapes)}")
    print(f"Test shapes: {len(test_shapes)}")
    
    # Create datasets
    trainset = MCBCoralDataset(
        data_root,
        train_shapes,
        split="train",
        latent_dim=latent_dim,
        normalize=True,
        subsample_interior=subsample_interior,
    )
    
    testset = MCBCoralDataset(
        data_root,
        test_shapes,
        split="test",
        latent_dim=latent_dim,
        normalize=True,
        stats=trainset.stats,
        subsample_interior=subsample_interior,
    )
    
    ntrain = len(trainset)
    ntest = len(testset)
    
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        follow_batch=['input_pos', 'output_pos'],
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=['input_pos', 'output_pos'],
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    
    print(f"Train: {ntrain} samples, {len(train_loader)} batches")
    print(f"Test: {ntest} samples, {len(test_loader)} batches")
    
    # === Create INR models ===
    # 3D coordinates
    input_dim = 3
    output_dim_in = 1   # boundary values / source term
    output_dim_out = 1  # solution
    
    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_in, device="cuda"
    )
    
    cfg.inr = cfg.inr_out
    inr_out = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_out, device="cuda"
    )
    
    print(f"\nInput INR params: {sum(p.numel() for p in inr_in.parameters()):,}")
    print(f"Output INR params: {sum(p.numel() for p in inr_out.parameters()):,}")
    
    # Learnable inner loop learning rates
    alpha_in = nn.Parameter(torch.tensor([lr_code], device="cuda"))
    alpha_out = nn.Parameter(torch.tensor([lr_code], device="cuda"))
    
    # === Optimizers ===
    optimizer_in = torch.optim.AdamW(
        [
            {"params": inr_in.parameters()},
            {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )
    
    optimizer_out = torch.optim.AdamW(
        [
            {"params": inr_out.parameters()},
            {"params": alpha_out, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )
    
    # Learning rate schedulers
    scheduler_in = torch.optim.lr_scheduler.StepLR(
        optimizer_in, step_size=100, gamma=cfg.optim.gamma_step
    )
    scheduler_out = torch.optim.lr_scheduler.StepLR(
        optimizer_out, step_size=100, gamma=cfg.optim.gamma_step
    )
    
    # === Training loop ===
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    best_loss = np.inf

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        fit_train_mse_in = 0.0
        fit_train_mse_out = 0.0
        fit_test_mse_in = 0.0
        fit_test_mse_out = 0.0
        
        use_rel_loss = (epoch % 10 == 0)
        do_test = (epoch % 20 == 0) or (epoch == epochs - 1)
        
        # === Training ===
        inr_in.train()
        inr_out.train()
        
        for batch_idx, (graph, idx) in enumerate(train_loader):
            n_samples = graph.num_graphs
            graph = graph.cuda()
            
            # --- Train Input INR (boundary values + source term) ---
            input_coords = graph.input_pos      # (total_N, 3)
            input_features = graph.input_values  # (total_N, 1)
            input_batch = graph.input_pos_batch
            
            outputs_in = graph_outer_step_mcb(
                inr_in,
                input_coords,
                input_features,
                input_batch,
                n_samples,
                latent_dim,
                inner_steps,
                alpha_in,
                is_train=True,
                use_rel_loss=use_rel_loss,
            )
            
            optimizer_in.zero_grad()
            outputs_in["loss"].backward()
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
            optimizer_in.step()
            
            fit_train_mse_in += outputs_in["loss"].item() * n_samples
            
            # --- Train Output INR (solution) ---
            output_coords = graph.output_pos      # (total_N, 3)
            output_features = graph.output_values  # (total_N, 1)
            output_batch = graph.output_pos_batch
            
            outputs_out = graph_outer_step_mcb(
                inr_out,
                output_coords,
                output_features,
                output_batch,
                n_samples,
                latent_dim,
                inner_steps,
                alpha_out,
                is_train=True,
                use_rel_loss=use_rel_loss,
            )
            
            optimizer_out.zero_grad()
            outputs_out["loss"].backward()
            nn.utils.clip_grad_value_(inr_out.parameters(), clip_value=1.0)
            optimizer_out.step()
            
            fit_train_mse_out += outputs_out["loss"].item() * n_samples
        
        scheduler_in.step()
        scheduler_out.step()
        
        train_loss_in = fit_train_mse_in / ntrain
        train_loss_out = fit_train_mse_out / ntrain
        
        # === Testing ===
        if do_test:
            inr_in.eval()
            inr_out.eval()
            
            with torch.no_grad():
                for batch_idx, (graph, idx) in enumerate(test_loader):
                    n_samples = graph.num_graphs
                    graph = graph.cuda()
                    
                    # Test Input INR
                    input_coords = graph.input_pos
                    input_features = graph.input_values
                    input_batch = graph.input_pos_batch
                    
                    outputs_in = graph_outer_step_mcb(
                        inr_in,
                        input_coords,
                        input_features,
                        input_batch,
                        n_samples,
                        latent_dim,
                        test_inner_steps,
                        alpha_in,
                        is_train=False,
                    )
                    fit_test_mse_in += outputs_in["loss"].item() * n_samples
                    
                    # Test Output INR
                    output_coords = graph.output_pos
                    output_features = graph.output_values
                    output_batch = graph.output_pos_batch
                    
                    outputs_out = graph_outer_step_mcb(
                        inr_out,
                        output_coords,
                        output_features,
                        output_batch,
                        n_samples,
                        latent_dim,
                        test_inner_steps,
                        alpha_out,
                        is_train=False,
                    )
                    fit_test_mse_out += outputs_out["loss"].item() * n_samples
            
            test_loss_in = fit_test_mse_in / ntest
            test_loss_out = fit_test_mse_out / ntest
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}: Train(in={train_loss_in:.6f}, out={train_loss_out:.6f}) "
                  f"Test(in={test_loss_in:.6f}, out={test_loss_out:.6f}) "
                  f"α_in={alpha_in.item():.4f}, α_out={alpha_out.item():.4f} "
                  f"[{epoch_time:.1f}s]")
        else:
            test_loss_in = 0.0
            test_loss_out = 0.0
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}: Train(in={train_loss_in:.6f}, out={train_loss_out:.6f}) "
                  f"α_in={alpha_in.item():.4f}, α_out={alpha_out.item():.4f} "
                  f"[{epoch_time:.1f}s]")
        
        # === Logging ===
        wandb.log(
            {
                "train_loss_in": train_loss_in,
                "train_loss_out": train_loss_out,
                "test_loss_in": test_loss_in,
                "test_loss_out": test_loss_out,
                "alpha_in": alpha_in.item(),
                "alpha_out": alpha_out.item(),
                "lr_in": scheduler_in.get_last_lr()[0],
                "lr_out": scheduler_out.get_last_lr()[0],
                "epoch_time": epoch_time,
            },
            step=epoch,
        )
        
        # === Checkpointing ===
        if train_loss_out < best_loss:
            best_loss = train_loss_out
            
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": epoch,
                    "inr_in": inr_in.state_dict(),
                    "inr_out": inr_out.state_dict(),
                    "optimizer_in": optimizer_in.state_dict(),
                    "optimizer_out": optimizer_out.state_dict(),
                    "loss": test_loss_out,
                    "alpha_in": alpha_in,
                    "alpha_out": alpha_out,
                    "stats": trainset.stats,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )
            print(f"  → Saved checkpoint (best loss: {best_loss:.6f})")
    
    # Final save
    torch.save(
        {
            "cfg": cfg,
            "epoch": epochs - 1,
            "inr_in": inr_in.state_dict(),
            "inr_out": inr_out.state_dict(),
            "optimizer_in": optimizer_in.state_dict(),
            "optimizer_out": optimizer_out.state_dict(),
            "loss": test_loss_out,
            "alpha_in": alpha_in,
            "alpha_out": alpha_out,
            "stats": trainset.stats,
        },
        f"{RESULTS_DIR}/{run_name}_final.pt",
    )
    
    wandb.finish()
    return test_loss_out


if __name__ == "__main__":
    main()