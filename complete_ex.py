if __name__ == '__main__':
    import warnings

    # Ignore specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import torch
    import torch_geometric.data as tg_data
    from torch_geometric.loader import DataLoader
    from MolecularGVAE import MolecularGVAE
    from MolecularGVAEModule import MolecularGVAEModule, train_gvae

    # First, create your base model (MolecularGVAE)
    model_params = {
        'node_features': 27,
        'hidden_dim': 64,
        'latent_dim': 32,
        'max_nodes': 500
    }

    # Initialize the base GVAE model
    base_model = MolecularGVAE(**model_params)

    # Wrap it in the Lightning module for training
    lightning_model = MolecularGVAEModule(
        model=base_model,
        lr=0.001,
        beta=0.1
    )

    # Create example dataset
    class MoleculeDataset(tg_data.Dataset):
        def __init__(self, data_list):
            super().__init__()
            self.data = data_list

        def len(self):
            return len(self.data)

        def get(self, idx):
            return self.data[idx]

    # Create some example graphs of different sizes
    data_list = []
    for size in range(20, 51, 5):  # Creates graphs with 20, 25, 30, ... 50 nodes
        # Random node features
        x = torch.randn(size, 27)

        # Create some random edges (you would replace this with your actual edges)
        num_edges = size * 2  # arbitrary number of edges
        edge_index = torch.randint(0, size, (2, num_edges))

        # Create PyG Data object
        data = tg_data.Data(
            x=x,
            edge_index=edge_index
        )
        data_list.append(data)

    # Split into train/val
    train_data = data_list[:8]
    val_data = data_list[8:]

    train_dataset = MoleculeDataset(train_data)
    val_dataset = MoleculeDataset(val_data)

    # Setup training
    training_params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 10,
        'num_workers': 0,
        'gradient_clip_val': 1.0,
        'beta': 0.1
    }

    # Train using the pipeline
    model, trainer = train_gvae(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_params=model_params,
        training_params=training_params,
    )

