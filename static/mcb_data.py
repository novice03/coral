"""
CORAL Dataset for MCB Multi-Shape Poisson Problem

Task: Given boundary conditions + source term → predict solution

Input INR architecture:
- Coordinates: boundary_points ∪ interior_coords (M + N_interior, 3)
- Values: boundary_values ∪ source_term[interior] (M + N_interior, 1)
- Learns latent code z_in encoding both BC and source information

Output INR architecture:
- Coordinates: all mesh_nodes (N, 3)
- Values: solution (N, 1)
- Learns latent code z_out

Mapping: z_in → z_out via ResNet (no explicit conditioning)
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import glob


def extract_shape_ids_from_file(filepath):
    """Extract unique shape IDs from NGF data list file."""
    shape_ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('/')
            for i, part in enumerate(parts):
                if part == 'mcb_b' and i + 2 < len(parts):
                    shape_ids.add(parts[i + 2])
                    break
    return sorted(shape_ids)


class MCBCoralDataset(Dataset):
    """
    PyTorch Geometric Dataset for CORAL on MCB Poisson problem.
    
    Each sample contains:
    - Input field: boundary_points + interior_coords → boundary_values + source_term[interior]
    - Output field: all mesh_nodes → solution
    
    Data loaded from .npz files with structure:
    - v_tet: (N, 3) vertex coordinates
    - source_term: (N,) or (N, 1) source term values
    - sol: (N,) solution values
    - bd_v_inds: (M,) boundary vertex indices
    - bd_v_vals: (M,) boundary values
    """

    def __init__(
        self,
        data_root: str,
        shape_ids: List[str],
        split: str = "train",
        latent_dim: int = 256,
        normalize: bool = True,
        stats: Optional[Dict] = None,
        subsample_interior: Optional[int] = None,
        cache_dir: str = "cache",
        use_cache: bool = True,
        preload: bool = True,
    ):
        """
        Args:
            data_root: Root directory containing shape subdirectories
            shape_ids: List of shape IDs to include
            split: One of "train", "test" (for logging purposes)
            latent_dim: Dimension of INR modulation codes
            normalize: Whether to normalize data
            stats: Pre-computed normalization stats (use training stats for test)
            subsample_interior: If set, randomly subsample interior points
            cache_dir: Directory for caching stats
            use_cache: Whether to use cached stats
            preload: Whether to preload all data into RAM
        """
        super().__init__(None, None, None)
        
        self.data_root = Path(data_root)
        self.split = split
        self.latent_dim = latent_dim
        self.normalize = normalize
        self.subsample_interior = subsample_interior
        self.stats = stats
        self.preload = preload
        
        # Collect all sample files from specified shapes
        self.file_paths = []
        self.shape_to_files = {}
        
        for shape_id in shape_ids:
            shape_path = self.data_root / shape_id
            if not shape_path.exists():
                print(f"Warning: Shape directory not found: {shape_path}")
                continue
            
            sample_files = sorted([
                f for f in glob.glob(str(shape_path / '*.npz'))
                if not f.endswith('mass_matrix.npz')
            ])
            self.shape_to_files[shape_id] = sample_files
            self.file_paths.extend(sample_files)
        
        print(f"[{split}] Loaded {len(self.file_paths)} samples from {len(self.shape_to_files)} shapes")
        
        if len(self.file_paths) == 0:
            raise ValueError("No valid files found! Check paths.")
        
        # Compute or load normalization stats
        if self.normalize and self.stats is None:
            os.makedirs(cache_dir, exist_ok=True)
            stats_file = os.path.join(
                cache_dir,
                f"mcb_coral_stats_{split}_{len(self.file_paths)}.pkl"
            )
            
            if use_cache and os.path.exists(stats_file):
                print(f"Loading stats from: {stats_file}")
                with open(stats_file, 'rb') as f:
                    self.stats = pickle.load(f)
            else:
                self._compute_and_cache_stats(stats_file)
        
        # Preload data
        self.dataset = {}
        if preload:
            self._preload_all_data()
    
    def _compute_and_cache_stats(self, cache_file: str):
        """Compute normalization statistics from a subset of data."""
        print("Computing normalization statistics...")
        
        all_coords = []
        all_source = []
        all_sol = []
        all_bd_vals = []
        
        # Sample up to 200 files spread across shapes
        n_sample = min(200, len(self.file_paths))
        sample_indices = np.linspace(0, len(self.file_paths) - 1, n_sample, dtype=int)
        sample_files = [self.file_paths[i] for i in sample_indices]
        
        for fpath in tqdm(sample_files, desc="Computing stats"):
            try:
                data = np.load(fpath)
                all_coords.append(data['v_tet'])
                source = data['source_term']
                if source.ndim > 1:
                    source = source.flatten()
                all_source.append(source)
                all_sol.append(data['sol'])
                all_bd_vals.append(data['bd_v_vals'])
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue
        
        if len(all_coords) == 0:
            raise ValueError("No valid files found for computing statistics!")
        
        coords = np.concatenate(all_coords, axis=0)
        source = np.concatenate(all_source, axis=0)
        sol = np.concatenate(all_sol, axis=0)
        bd_vals = np.concatenate(all_bd_vals, axis=0)
        
        self.stats = {
            'coords_mean': coords.mean(axis=0).astype(np.float32),
            'coords_std': (coords.std(axis=0) + 1e-8).astype(np.float32),
            'source_mean': float(source.mean()),
            'source_std': float(source.std() + 1e-8),
            'sol_mean': float(sol.mean()),
            'sol_std': float(sol.std() + 1e-8),
            'bd_vals_mean': float(bd_vals.mean()),
            'bd_vals_std': float(bd_vals.std() + 1e-8),
        }
        
        # For the input INR, we combine boundary values and source term
        # Compute combined stats for the "input values"
        # We'll normalize them separately but track for reference
        all_input_vals = np.concatenate([source, bd_vals])
        self.stats['input_vals_mean'] = float(all_input_vals.mean())
        self.stats['input_vals_std'] = float(all_input_vals.std() + 1e-8)
        
        print(f"Stats computed:")
        print(f"  Coords: mean={self.stats['coords_mean']}, std={self.stats['coords_std']}")
        print(f"  Source: mean={self.stats['source_mean']:.4f}, std={self.stats['source_std']:.4f}")
        print(f"  Solution: mean={self.stats['sol_mean']:.4f}, std={self.stats['sol_std']:.4f}")
        print(f"  BD vals: mean={self.stats['bd_vals_mean']:.4f}, std={self.stats['bd_vals_std']:.4f}")
        print(f"Saving to: {cache_file}")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.stats, f)
    
    def _preload_all_data(self):
        """Preload all data into memory as PyG Data objects."""
        print(f"Preloading {len(self.file_paths)} samples...")
        
        for local_idx, fpath in enumerate(tqdm(self.file_paths, desc="Loading")):
            try:
                raw_data = np.load(fpath)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue
            
            # Extract raw data
            coords = raw_data['v_tet'].astype(np.float32)  # (N, 3)
            source_term = raw_data['source_term'].astype(np.float32)
            if source_term.ndim > 1:
                source_term = source_term.flatten()  # (N,)
            solution = raw_data['sol'].astype(np.float32)  # (N,)
            bd_v_inds = raw_data['bd_v_inds'].astype(np.int64)  # (M,)
            bd_v_vals = raw_data['bd_v_vals'].astype(np.float32)  # (M,)
            
            n_vertices = len(coords)
            
            # Create boundary mask
            is_boundary = np.zeros(n_vertices, dtype=bool)
            is_boundary[bd_v_inds] = True
            interior_mask = ~is_boundary
            
            # Get interior indices
            interior_inds = np.where(interior_mask)[0]
            
            # Subsample interior if requested
            if self.subsample_interior is not None and len(interior_inds) > self.subsample_interior:
                np.random.seed(local_idx)  # Reproducible subsampling
                interior_inds = np.random.choice(
                    interior_inds,
                    size=self.subsample_interior,
                    replace=False
                )
                interior_inds = np.sort(interior_inds)
            
            # === Build Input INR data ===
            # Boundary points and their BC values
            boundary_coords = coords[bd_v_inds]  # (M, 3)
            boundary_values = bd_v_vals  # (M,)
            
            # Input INR only uses boundary data
            input_coords = boundary_coords  # (M, 3)
            input_values = boundary_values  # (M,)
            
            # Track which points are boundary vs interior in input
            n_boundary = len(bd_v_inds)
            n_interior = len(interior_inds)
            
            # === Build Output INR data ===
            # Use all mesh nodes (or subsampled version)
            if self.subsample_interior is not None:
                # Keep all boundary + subsampled interior
                output_inds = np.concatenate([bd_v_inds, interior_inds])
                output_inds = np.sort(output_inds)
                output_coords = coords[output_inds]
                output_solution = solution[output_inds]
            else:
                output_coords = coords
                output_solution = solution
            
            # === Apply normalization ===
            if self.normalize:
                # Normalize coordinates (same for input and output)
                input_coords = (input_coords - self.stats['coords_mean']) / self.stats['coords_std']
                output_coords = (output_coords - self.stats['coords_mean']) / self.stats['coords_std']
                
                # Normalize input values (boundary values only)
                input_values = (input_values - self.stats['bd_vals_mean']) / self.stats['bd_vals_std']
                
                # Normalize output solution
                output_solution = (output_solution - self.stats['sol_mean']) / self.stats['sol_std']
            
            # === Create PyG Data object ===
            data = Data()
            
            # Input INR fields
            data.input_pos = torch.from_numpy(input_coords).float()  # (M, 3)
            data.input_values = torch.from_numpy(input_values).float().unsqueeze(-1)  # (M, 1)
            data.num_input_nodes = len(input_coords)
            
            # Output INR fields
            data.output_pos = torch.from_numpy(output_coords).float()  # (N_out, 3)
            data.output_values = torch.from_numpy(output_solution).float().unsqueeze(-1)  # (N_out, 1)
            data.num_output_nodes = len(output_coords)
            
            # Metadata
            data.n_boundary = n_boundary
            data.n_interior = n_interior
            data.file_path = fpath
            
            # Latent codes (initialized to zero, fitted during training)
            data.z_input = torch.zeros(1, self.latent_dim)
            data.z_output = torch.zeros(1, self.latent_dim)
            
            # For CORAL compatibility
            data.z_vx = torch.zeros(1, self.latent_dim, 2)
            
            self.dataset[str(local_idx)] = data
        
        print(f"✓ Loaded {len(self.dataset)} samples")
        
        # Print sample info
        if len(self.dataset) > 0:
            sample = self.dataset['0']
            print(f"\nSample structure:")
            print(f"  input_pos: {sample.input_pos.shape}")
            print(f"  input_values: {sample.input_values.shape}")
            print(f"  output_pos: {sample.output_pos.shape}")
            print(f"  output_values: {sample.output_values.shape}")
            print(f"  n_boundary: {sample.n_boundary}, n_interior: {sample.n_interior}")
    
    def len(self) -> int:
        return len(self.dataset)
    
    def indices(self) -> list:
        return list(range(len(self.dataset)))
    
    def get(self, idx: int) -> Tuple[Data, int]:
        """Get a sample in CORAL-compatible format."""
        graph = self.dataset[str(idx)].clone()
        return (graph, idx)
    
    def set_code(self, z_values: torch.Tensor, key: str):
        """Store fitted modulation codes."""
        z_values = z_values.cpu().detach()
        for index in range(self.len()):
            if key == "input":
                self.dataset[str(index)].z_vx[..., 0] = z_values[index]
            elif key == "output":
                self.dataset[str(index)].z_vx[..., 1] = z_values[index]


class MCBCoralDataEncapsulator:
    """
    Data loading encapsulator for MCB CORAL training.
    
    Handles train/test split based on NGF data list files.
    """
    
    def __init__(
        self,
        data_root: str,
        train_split_file: str,
        test_split_file: str,
        latent_dim: int = 256,
        batch_size: int = 4,
        normalize: bool = True,
        subsample_interior: Optional[int] = 2000,
        num_workers: int = 4,
        cache_dir: str = "cache",
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing shape subdirectories
            train_split_file: Path to file listing train shapes
            test_split_file: Path to file listing test shapes
            latent_dim: Dimension of INR modulation codes
            batch_size: Batch size for DataLoaders
            normalize: Whether to normalize data
            subsample_interior: If set, randomly subsample interior points
            num_workers: Number of workers for DataLoaders
            cache_dir: Directory for caching stats
        """
        from torch_geometric.loader import DataLoader
        
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        print(f"\n{'='*70}")
        print(f"Loading MCB CORAL Dataset")
        print(f"{'='*70}")
        print(f"Data root: {data_root}")
        
        # Extract shape IDs from split files
        train_shapes = extract_shape_ids_from_file(train_split_file)
        test_shapes = extract_shape_ids_from_file(test_split_file)
        
        # Verify shapes exist
        existing_shapes = set(os.listdir(self.data_root))
        train_shapes = [s for s in train_shapes if s in existing_shapes]
        test_shapes = [s for s in test_shapes if s in existing_shapes]
        
        print(f"\nTrain shapes: {len(train_shapes)}")
        print(f"Test shapes: {len(test_shapes)}")
        
        # Create datasets
        print("\nCreating training dataset...")
        self.train_dataset = MCBCoralDataset(
            data_root,
            train_shapes,
            split="train",
            latent_dim=latent_dim,
            normalize=normalize,
            subsample_interior=subsample_interior,
            cache_dir=cache_dir,
        )
        
        # Use training stats for test set
        test_stats = self.train_dataset.stats if normalize else None
        
        print("\nCreating test dataset...")
        self.test_dataset = MCBCoralDataset(
            data_root,
            test_shapes,
            split="test",
            latent_dim=latent_dim,
            normalize=normalize,
            stats=test_stats,
            subsample_interior=subsample_interior,
            cache_dir=cache_dir,
        )
        
        self.stats = self.train_dataset.stats
        
        # Create DataLoaders
        print(f"\nCreating DataLoaders...")
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            follow_batch=['input_pos', 'output_pos'],
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            follow_batch=['input_pos', 'output_pos'],
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        print(f"\nDataLoaders ready:")
        print(f"  Training: {len(self.train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"  Testing: {len(self.test_dataset)} samples, {len(self.test_loader)} batches")
    
    def decode_solution(self, sol_normalized):
        """Denormalize solution."""
        if self.stats is not None:
            mean = self.stats['sol_mean']
            std = self.stats['sol_std']
            return sol_normalized * std + mean
        return sol_normalized


# Test
if __name__ == "__main__":
    print("Testing MCB CORAL Data Loader...")
    
    data = MCBCoralDataEncapsulator(
        data_root='/export/mkrishn9/iae_net/datasets/ngf/data_poisson_multishape_20260126_085543',
        train_split_file='/export/mkrishn9/NGF/data_list_paths/mcb_b/fitting/known_shape_known_prob.txt',
        test_split_file='/export/mkrishn9/NGF/data_list_paths/mcb_b/fitting/unknown_shape_unknown_prob.txt',
        latent_dim=256,
        batch_size=2,
        subsample_interior=1000,
        num_workers=0,
    )
    
    # Test train loader
    print("\n" + "="*70)
    print("Testing train loader...")
    for batch, idx in data.train_loader:
        print("\nBatch attributes:")
        print(f"  input_pos: {batch.input_pos.shape}")
        print(f"  input_values: {batch.input_values.shape}")
        print(f"  input_pos_batch: {batch.input_pos_batch.shape}")
        print(f"  output_pos: {batch.output_pos.shape}")
        print(f"  output_values: {batch.output_values.shape}")
        print(f"  output_pos_batch: {batch.output_pos_batch.shape}")
        print(f"  num_graphs: {batch.num_graphs}")
        break
    
    print("\n✓ Data loader test complete!")