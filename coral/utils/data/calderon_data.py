"""
CORAL Dataset for Inverse Calderón Problem (Electrical Impedance Tomography)

Task: Map boundary measurements (Dirichlet-Neumann pairs) → conductivity on mesh

Architecture:
- Input INR: Maps sensor coordinates (x, y) → 40 values (20 Dirichlet + 20 Neumann stacked)
- Output INR: Maps mesh coordinates (x, y) → 1 value (conductivity)
- Mapping: z_input → z_output via ResNet/MLP

Key insight: CORAL uses MAML-style meta-learning to find modulations (latent codes)
that condition the INR to fit each sample. The coordinates are the "queries" and
the field values are the "targets".
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class CalderonCoralDataset(Dataset):
    """
    PyTorch Geometric Dataset for CORAL on Calderón/EIT inverse problem.
    
    Each sample contains:
    - Input field: At M sensor locations, we have L=20 Dirichlet + L=20 Neumann values
      → INR maps (x, y) → 40 values
    - Output field: At N mesh nodes, we have conductivity
      → INR maps (x, y) → 1 value
    
    The key is that input and output have DIFFERENT coordinate systems:
    - Input coordinates: sensor_locations (M points on boundary)
    - Output coordinates: mesh_nodes (N points in domain)
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        latent_dim: int = 128,
        normalize: bool = True,
        cache_dir: str = "cache",
        use_cache: bool = True,
        stats: Optional[Dict] = None,
        ntrain: int = 400,
        ntest: int = 100,
        nval: int = 50,
        seed: int = 42,
        preload: bool = True,
        # Subsampling parameters
        subsample_mesh: bool = False,
        subsample_size: int = 1024,
        subsample_seed: Optional[int] = None,  # If set, use fixed seed for reproducible subsampling
    ):
        """
        Args:
            data_path: Path to calderon_pentagon_dataset.mat
            split: One of "train", "val", "test"
            latent_dim: Dimension of INR modulation codes
            normalize: Whether to normalize data
            cache_dir: Directory for caching stats
            use_cache: Whether to use cached stats
            stats: Pre-computed normalization stats (for val/test sets)
            ntrain, ntest, nval: Number of samples per split
            seed: Random seed for splitting
            preload: Whether to preload all data into RAM
            subsample_mesh: Whether to subsample mesh nodes (for faster training)
            subsample_size: Number of mesh nodes to sample per sample
            subsample_seed: If set, use fixed random seed for subsampling (for reproducible eval)
        """
        super().__init__(None, None, None)
        
        self.data_path = data_path
        self.split = split
        self.latent_dim = latent_dim
        self.normalize = normalize
        self.preload = preload
        self.stats = stats
        
        # Subsampling settings
        self.subsample_mesh = subsample_mesh
        self.subsample_size = subsample_size
        self.subsample_seed = subsample_seed
        
        # Determine indices for this split
        np.random.seed(seed)
        with h5py.File(data_path, 'r') as f:
            total_samples = f['DataSet']['geometry_params'].shape[0]
        
        all_indices = np.random.permutation(total_samples)
        
        if split == "train":
            self._indices = all_indices[:ntrain].tolist()
        elif split == "val":
            self._indices = all_indices[ntrain:ntrain + nval].tolist()
        elif split == "test":
            self._indices = all_indices[ntrain + nval:ntrain + nval + ntest].tolist()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"[{split}] Using {len(self._indices)} samples")
        if subsample_mesh:
            print(f"  Mesh subsampling ENABLED: {subsample_size} nodes per sample")
        
        # Get data dimensions
        with h5py.File(data_path, 'r') as h5file:
            first_idx = self._indices[0]
            dirichlet_ref = h5file['DataSet']['dirichlet_data'][first_idx, 0]
            dirichlet_sample = h5file[dirichlet_ref][()]
            # Data shape is (M, L) where M=272 sensors, L=20 boundary conditions
            self.M, self.L = dirichlet_sample.shape  # M=272 sensors, L=20 boundary conditions
            
            # Get mesh size for first sample
            nodes_ref = h5file['DataSet']['mesh_nodes'][first_idx, 0]
            nodes_sample = h5file[nodes_ref][()]
            if nodes_sample.shape[0] == 2:
                self.N_example = nodes_sample.shape[1]
            else:
                self.N_example = nodes_sample.shape[0]
        
        print(f"Data dimensions: M={self.M} sensors, L={self.L} BCs, ~N={self.N_example} mesh nodes")
        
        # Compute or load normalization stats
        if self.normalize and self.stats is None:
            os.makedirs(cache_dir, exist_ok=True)
            stats_file = os.path.join(
                cache_dir,
                f"calderon_coral_stats_{split}_{len(self._indices)}.pkl"
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
        """Compute normalization statistics."""
        print("Computing normalization statistics...")
        
        all_dirichlet = []
        all_neumann = []
        all_conductivity = []
        all_sensor_coords = []
        all_mesh_coords = []
        
        with h5py.File(self.data_path, 'r') as h5file:
            dataset = h5file['DataSet']
            
            for idx in tqdm(self._indices, desc="Computing stats"):
                # Dirichlet data (L, M)
                dirich_ref = dataset['dirichlet_data'][idx, 0]
                dirich = h5file[dirich_ref][()].astype(np.float32)
                all_dirichlet.append(dirich.flatten())
                
                # Neumann data (L, M)
                neum_ref = dataset['neumann_data'][idx, 0]
                neum = h5file[neum_ref][()].astype(np.float32)
                all_neumann.append(neum.flatten())
                
                # Conductivity
                cond_ref = dataset['conductivity_on_mesh'][idx, 0]
                cond = h5file[cond_ref][()].flatten().astype(np.float32)
                all_conductivity.append(cond)
                
                # Sensor locations (2, M) -> flatten
                sensor_ref = dataset['sensor_locations'][idx, 0]
                sensors = h5file[sensor_ref][()].astype(np.float32)
                if sensors.shape[0] != 2:
                    sensors = sensors.T
                all_sensor_coords.append(sensors.flatten())
                
                # Mesh nodes
                nodes_ref = dataset['mesh_nodes'][idx, 0]
                nodes = h5file[nodes_ref][()].astype(np.float32)
                if nodes.shape[0] == 2:
                    nodes = nodes.T
                all_mesh_coords.append(nodes.flatten())
        
        # Compute statistics
        all_dirichlet = np.concatenate(all_dirichlet)
        all_neumann = np.concatenate(all_neumann)
        all_conductivity = np.concatenate(all_conductivity)
        all_sensor_coords = np.concatenate(all_sensor_coords)
        all_mesh_coords = np.concatenate(all_mesh_coords)
        
        self.stats = {
            'dirichlet_mean': np.mean(all_dirichlet),
            'dirichlet_std': np.std(all_dirichlet) + 1e-8,
            'neumann_mean': np.mean(all_neumann),
            'neumann_std': np.std(all_neumann) + 1e-8,
            'conductivity_mean': np.mean(all_conductivity),
            'conductivity_std': np.std(all_conductivity) + 1e-8,
            'sensor_mean': np.mean(all_sensor_coords),
            'sensor_std': np.std(all_sensor_coords) + 1e-8,
        }
        
        print(f"Stats computed. Saving to: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.stats, f)
    
    def _preload_all_data(self):
        """Preload all data into memory as PyG Data objects."""
        print(f"Preloading {len(self._indices)} samples...")
        
        with h5py.File(self.data_path, 'r') as h5file:
            dataset = h5file['DataSet']
            
            for local_idx, global_idx in enumerate(tqdm(self._indices, desc="Loading")):
                # Load sensor locations: (2, M) -> (M, 2)
                sensor_ref = dataset['sensor_locations'][global_idx, 0]
                sensor_coords = h5file[sensor_ref][()].astype(np.float32)
                if sensor_coords.shape[0] == 2:
                    sensor_coords = sensor_coords.T  # (M, 2)
                
                # Load Dirichlet data: (L, M)
                dirich_ref = dataset['dirichlet_data'][global_idx, 0]
                dirichlet = h5file[dirich_ref][()].astype(np.float32)  # (L, M)
                
                # Load Neumann data: (L, M)
                neum_ref = dataset['neumann_data'][global_idx, 0]
                neumann = h5file[neum_ref][()].astype(np.float32)  # (L, M)
                
                # Load mesh nodes: (2, N) -> (N, 2)
                nodes_ref = dataset['mesh_nodes'][global_idx, 0]
                mesh_coords = h5file[nodes_ref][()].astype(np.float32)
                if mesh_coords.shape[0] == 2:
                    mesh_coords = mesh_coords.T  # (N, 2)
                
                # Load conductivity: (N,)
                cond_ref = dataset['conductivity_on_mesh'][global_idx, 0]
                conductivity = h5file[cond_ref][()].flatten().astype(np.float32)
                
                # Load geometry params (optional, for conditioning)
                geo_ref = dataset['geometry_params'][global_idx, 0]
                geo_params = h5file[geo_ref][()].flatten().astype(np.float32)
                geo_params = geo_params[:5] if len(geo_params) > 5 else geo_params
                
                # Apply normalization
                if self.normalize:
                    dirichlet = (dirichlet - self.stats['dirichlet_mean']) / self.stats['dirichlet_std']
                    neumann = (neumann - self.stats['neumann_mean']) / self.stats['neumann_std']
                    conductivity = (conductivity - self.stats['conductivity_mean']) / self.stats['conductivity_std']
                    sensor_coords = (sensor_coords - self.stats['sensor_mean']) / self.stats['sensor_std']
                
                # Stack Dirichlet and Neumann: both are (M, L) shape
                # Concatenate along axis=1: (M, L) + (M, L) -> (M, 2L)
                # This gives us 40 values (20 Dirichlet + 20 Neumann) per sensor location
                input_values = np.concatenate([dirichlet, neumann], axis=1)  # (M, 2L=40)
                
                M = sensor_coords.shape[0]  # Number of sensors
                N = mesh_coords.shape[0]    # Number of mesh nodes
                
                # Create PyG Data object
                data = Data()
                
                # Input space (sensor measurements)
                data.input_pos = torch.from_numpy(sensor_coords).float()  # (M, 2) - input coordinates
                data.input_values = torch.from_numpy(input_values).float()  # (M, 40) - input field values
                data.num_input_nodes = M  # Track for batching
                
                # Output space (conductivity on mesh)
                data.output_pos = torch.from_numpy(mesh_coords).float()  # (N, 2) - output coordinates
                data.output_values = torch.from_numpy(conductivity).float().unsqueeze(-1)  # (N, 1)
                data.num_output_nodes = N  # Track for batching
                
                # Store dimensions
                data.M = M
                data.N = N
                
                # Geometry params (optional)
                data.geo_params = torch.from_numpy(geo_params).float()
                
                # Latent codes (initialized to zero, will be fitted during training)
                # z_input: modulation for input INR
                # z_output: modulation for output INR
                data.z_input = torch.zeros(1, self.latent_dim)
                data.z_output = torch.zeros(1, self.latent_dim)
                
                # For CORAL compatibility - z_vx[..., 0] = input code, z_vx[..., 1] = output code
                data.z_vx = torch.zeros(1, self.latent_dim, 2)
                
                # Store original N for subsampling
                data.N_original = N
                
                # Store global index for reproducible subsampling
                data.global_idx = global_idx
                
                self.dataset[str(local_idx)] = data
        
        print(f"✓ Loaded {len(self.dataset)} samples")
    
    def len(self) -> int:
        return len(self._indices)
    
    def indices(self) -> list:
        """Override to return local indices (0 to len-1) instead of global HDF5 indices."""
        return list(range(len(self._indices)))
    
    def get(self, idx: int) -> Tuple[Data, int]:
        """
        Get a sample in CORAL-compatible format.
        
        Note: PyG Dataset calls get() with indices from 0 to len()-1.
        The self._indices list maps these local indices to global HDF5 indices,
        but we store data with local indices in self.dataset.
        
        If subsampling is enabled, randomly subsample mesh nodes here.
        
        Returns graph with:
        - pos: coordinates for the CURRENT INR being trained
        - images: target values for the CURRENT INR
        - input/output: full data for reference
        - modulations: placeholder for latent codes
        """
        # idx is the local index (0, 1, 2, ..., len-1)
        graph = self.dataset[str(idx)].clone()
        
        # Apply mesh subsampling if enabled
        if self.subsample_mesh and graph.N_original > self.subsample_size:
            N_original = graph.N_original
            
            # Set up random number generator
            if self.subsample_seed is not None:
                # Reproducible subsampling (for eval)
                rng = np.random.RandomState(self.subsample_seed + graph.global_idx)
            else:
                # Random subsampling each time (for training)
                rng = np.random.RandomState()
            
            # Sample indices
            subsample_indices = rng.choice(
                N_original, 
                size=self.subsample_size, 
                replace=False
            )
            subsample_indices = np.sort(subsample_indices)
            
            # Apply subsampling to output (mesh) data
            graph.output_pos = graph.output_pos[subsample_indices]
            graph.output_values = graph.output_values[subsample_indices]
            graph.N = self.subsample_size
            graph.num_output_nodes = self.subsample_size
        
        # For CORAL's metalearning, we need:
        # - graph.pos: coordinates where INR is evaluated (N_points, coord_dim)
        # - graph.images: target values at those coordinates (N_points, value_dim)
        # - graph.modulations: latent codes (batch_size, latent_dim)
        
        # The training script will set these based on whether we're training input or output INR
        # Default: set up for input INR training
        graph.pos = graph.input_pos.unsqueeze(-1)  # (M, 2, 1) to match expected format
        graph.images = graph.input_values  # (M, 40)
        graph.modulations = graph.z_vx[..., 0]  # (1, latent_dim)
        
        # Also provide the "input" field for compatibility with static_inr.py
        # input = [pos, values] concatenated
        graph.input = torch.cat([graph.input_pos, graph.input_values], dim=-1)  # (M, 2+40=42)
        
        return (graph, idx)
    
    def set_code(self, z_values: torch.Tensor, key: str):
        """Store fitted modulation codes."""
        z_values = z_values.cpu().detach()
        for index in range(self.len()):
            if key == "input":
                self.dataset[str(index)].z_vx[..., 0] = z_values[index]
            elif key == "output":
                self.dataset[str(index)].z_vx[..., 1] = z_values[index]

