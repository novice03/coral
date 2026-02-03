"""
Static INR Training for Inverse Calderón Problem

This script trains two INRs:
1. Input INR: Maps sensor coordinates (x, y) → D/N values (40 channels)
2. Output INR: Maps mesh coordinates (x, y) → conductivity (1 channel)

The INRs use MAML-style meta-learning to find modulation codes that condition
the network to fit each individual sample.

Key difference from cylinder/airfoil: Input and output have DIFFERENT coordinate systems!
- Input: M=272 sensor locations on boundary
- Output: N~500 mesh nodes in domain interior
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

from coral.utils.models.load_inr import create_inr_instance
from coral.metalearning import graph_inner_loop_step, graph_inner_loop

from coral.utils.data.calderon_data import CalderonCoralDataset

def set_seed(seed=33):
    import random
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def graph_outer_step_calderon(
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
    loss_type="mse",
    detach_modulations=False,
    device="cuda",
):
    """
    Outer step for CORAL training adapted for Calderón problem.
    
    Unlike the original graph_outer_step, this takes coords/features/batch_index
    directly rather than extracting from graph object. This allows us to handle
    the case where input and output have different coordinate systems.
    
    Args:
        func_rep: INR model
        coords: (N_total, 2) coordinates
        features: (N_total, C) target values
        batch_index: (N_total,) which sample each point belongs to
        n_graphs: number of graphs in batch
        latent_dim: dimension of modulation codes
        inner_steps: number of inner loop steps
        inner_lr: inner loop learning rate (can be nn.Parameter)
        is_train: whether in training mode
        return_reconstructions: whether to return reconstructed values
        use_rel_loss: whether to compute relative loss
        loss_type: "mse" or other loss types
        detach_modulations: if True, detach modulations (first-order MAML)
        device: torch device
    """
    func_rep.zero_grad()
    
    # Initialize modulations to zero
    modulations = torch.zeros(n_graphs, latent_dim, device=device).requires_grad_()
    
    # Run inner loop to fit modulations
    for step in range(inner_steps):
        modulations = graph_inner_loop_step_calderon(
            func_rep,
            modulations,
            coords,
            features,
            batch_index,
            inner_lr,
            is_train,
        )
    
    if detach_modulations:
        modulations = modulations.detach()
    
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


def graph_inner_loop_step_calderon(
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
    
    Args:
        func_rep: INR model
        modulations: (batch_size, latent_dim) current modulations
        coords: (N_total, coord_dim) coordinates
        features: (N_total, feature_dim) target values
        batch_index: (N_total,) batch assignment for each point
        inner_lr: learning rate (scalar or nn.Parameter)
        is_train: if True, create computation graph for second-order gradients
    """
    batch_size = modulations.shape[0]
    
    # Ensure modulations requires grad
    if not modulations.requires_grad:
        modulations = modulations.requires_grad_()
    
    with torch.enable_grad():
        # Forward pass: reconstruct features
        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        
        # MSE loss, multiplied by batch_size to undo averaging
        # (each sample's modulations should be fit independently)
        loss = ((features_recon - features) ** 2).mean() * batch_size
        
        # Compute gradient w.r.t. modulations
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train,
        )[0]
    
    # Gradient descent step
    # The result needs requires_grad for subsequent inner loop steps
    new_modulations = modulations - inner_lr * grad
    
    # If not training (first-order MAML), we can detach and re-enable grad
    # If training (second-order MAML), the computation graph is already connected via create_graph=True
    if not is_train:
        new_modulations = new_modulations.detach().requires_grad_()
    
    return new_modulations


@hydra.main(config_path="config/calderon/", config_name="static_inr.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    torch.set_default_dtype(torch.float32)
    
    # === Extract config values ===
    # Data
    data_path = cfg.data.data_path
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    seed = cfg.data.seed
    
    # Optimization
    batch_size = cfg.optim.batch_size
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    
    # INR
    latent_dim = cfg.inr_in.latent_dim
    
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
    RESULTS_DIR = Path(os.getenv("WANDB_DIR", ".")) / "calderon" / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)
    
    # Set seed
    set_seed(seed)
    
    # Add tags
    dataset_name = cfg.data.dataset_name
    run.tags = ("calderon-inr", dataset_name, f"latent_dim={latent_dim}")
    
    # === Create datasets ===
    # Input INR: coords (2) → D/N values (40)
    # Output INR: coords (2) → conductivity (1)
    input_dim = 2
    output_dim_in = 40   # 20 Dirichlet + 20 Neumann
    output_dim_out = 1   # conductivity
    
    # Get subsampling config
    train_subsample = cfg.data.get('train_subsample_size', None)
    test_subsample = cfg.data.get('test_subsample_size', None)
    
    trainset = CalderonCoralDataset(
        data_path=data_path,
        split="train",
        latent_dim=latent_dim,
        normalize=True,
        ntrain=ntrain,
        ntest=ntest,
        seed=seed,
        subsample_mesh=(train_subsample is not None),
        subsample_size=train_subsample or 1024,
        subsample_seed=None,  # Random each time for training
    )
    
    testset = CalderonCoralDataset(
        data_path=data_path,
        split="test",
        latent_dim=latent_dim,
        normalize=True,
        stats=trainset.stats,  # Use training stats
        ntrain=ntrain,
        ntest=ntest,
        seed=seed,
        subsample_mesh=(test_subsample is not None),
        subsample_size=test_subsample or 1024,
        subsample_seed=42,  # Fixed for reproducible eval
    )
    
    ntrain = len(trainset)
    ntest = len(testset)
    
    # Use follow_batch to track batch indices for both coordinate systems
    train_loader = DataLoader(
        dataset=trainset, 
        batch_size=batch_size, 
        shuffle=True,
        follow_batch=['input_pos', 'output_pos'],
    )
    test_loader = DataLoader(
        dataset=testset, 
        batch_size=batch_size, 
        shuffle=False,
        follow_batch=['input_pos', 'output_pos'],
    )
    
    print(f"Train: {ntrain}, Test: {ntest}")
    
    # === Create INR models ===
    # Input INR
    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_in, device="cuda"
    )
    
    # Output INR
    cfg.inr = cfg.inr_out
    inr_out = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_out, device="cuda"
    )
    
    print(f"Input INR params: {sum(p.numel() for p in inr_in.parameters()):,}")
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
    
    # === Training loop ===
    best_loss = np.inf

    for epoch in range(epochs):
        # Metrics
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
            
            # --- Train Input INR ---
            # Coordinates: sensor locations (M points per sample)
            # Target: D/N values (40 channels)
            input_coords = graph.input_pos      # (total_M, 2)
            input_features = graph.input_values  # (total_M, 40)
            # PyG's follow_batch creates input_pos_batch when we specify follow_batch=['input_pos', ...]
            input_batch = graph.input_pos_batch
            
            outputs_in = graph_outer_step_calderon(
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
            z_in = outputs_in["modulations"].detach()
            
            # --- Train Output INR ---
            # Coordinates: mesh nodes (N points per sample, variable)
            # Target: conductivity (1 channel)
            output_coords = graph.output_pos      # (total_N, 2)
            output_features = graph.output_values  # (total_N, 1)
            # PyG's follow_batch creates output_pos_batch
            output_batch = graph.output_pos_batch
            
            outputs_out = graph_outer_step_calderon(
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
            z_out = outputs_out["modulations"].detach()
        
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
                    
                    # Input INR
                    input_coords = graph.input_pos
                    input_features = graph.input_values
                    input_batch = graph.input_pos_batch
                    
                    outputs_in = graph_outer_step_calderon(
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
                    
                    # Output INR
                    output_coords = graph.output_pos
                    output_features = graph.output_values
                    output_batch = graph.output_pos_batch
                    
                    outputs_out = graph_outer_step_calderon(
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
            
            print(f"Epoch {epoch}: Train(in={train_loss_in:.6f}, out={train_loss_out:.6f}) "
                  f"Test(in={test_loss_in:.6f}, out={test_loss_out:.6f}) "
                  f"alpha_in={alpha_in.item():.4f}, alpha_out={alpha_out.item():.4f}")
        else:
            test_loss_in = 0.0
            test_loss_out = 0.0
            print(f"Epoch {epoch}: Train(in={train_loss_in:.6f}, out={train_loss_out:.6f})")
        
        # === Logging ===
        wandb.log(
            {
                "train_loss_in": train_loss_in,
                "train_loss_out": train_loss_out,
                "test_loss_in": test_loss_in,
                "test_loss_out": test_loss_out,
                "alpha_in": alpha_in.item(),
                "alpha_out": alpha_out.item(),
            },
            step=epoch,
        )
        
        # === Checkpointing (use test loss for model selection) ===
        if do_test and test_loss_out < best_loss:
            best_loss = test_loss_out
            
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
                    "stats": trainset.stats,  # Save normalization stats
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )
            print(f"  → Saved checkpoint (best test loss: {best_loss:.6f})")
    
    wandb.finish()
    return test_loss_out


if __name__ == "__main__":
    main()