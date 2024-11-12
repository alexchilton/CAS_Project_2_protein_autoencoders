import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Batch
from torch_geometric.utils import to_dense_batch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from MolecularGVAE import MolecularGVAE
import os
import pytorch_lightning as pl
import torch


class MolecularGVAEModule(pl.LightningModule):
    def __init__(self, model, lr=0.001, beta=0.1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.beta = beta
        self.save_hyperparameters(ignore=['model'])

    def forward(self, batch):
        # Get batch information
        batch_size = batch.batch.max().item() + 1

        # Forward pass handling different sizes
        node_features, edge_logits, node_mu, node_logvar, graph_mu, graph_logvar = self.model(batch)

        return {
            'node_features': node_features,
            'edge_logits': edge_logits,
            'node_mu': node_mu,
            'node_logvar': node_logvar,
            'graph_mu': graph_mu,
            'graph_logvar': graph_logvar,
            'batch_size': batch_size
        }

    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch)

        # Calculate loss with batch information
        loss, metrics = self.model.loss_function(
            outputs['node_features'],
            outputs['edge_logits'],
            batch,
            outputs['node_mu'],
            outputs['node_logvar'],
            outputs['graph_mu'],
            outputs['graph_logvar']
        )

        # Calculate average loss per graph in batch
        loss = loss / outputs['batch_size']

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=outputs['batch_size'])
        for metric_name, metric_value in metrics.items():
            self.log(f'train_{metric_name}', metric_value, on_step=True, on_epoch=True,
                     batch_size=outputs['batch_size'])

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, metrics = self.model.loss_function(
            outputs['node_features'],
            outputs['edge_logits'],
            batch,
            outputs['node_mu'],
            outputs['node_logvar'],
            outputs['graph_mu'],
            outputs['graph_logvar']
        )

        loss = loss / outputs['batch_size']

        # Important: Log validation metrics properly
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=outputs['batch_size'],
                 sync_dist=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True, batch_size=outputs['batch_size'],
                     sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Modified scheduler configuration
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            ),
            "monitor": "train_loss",  # Changed from val_loss to train_loss
            "frequency": 1,
            "interval": "epoch"
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def train_gvae(train_dataset, val_dataset, model_params, training_params):
    # Initialize model
    model = MolecularGVAE(**model_params)

    # Create Lightning module
    lightning_model = MolecularGVAEModule(
        model=model,
        lr=training_params.get('learning_rate', 0.001),
        beta=training_params.get('beta', 0.1)
    )

    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params.get('batch_size', 32),
        shuffle=True,
        num_workers=training_params.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params.get('batch_size', 32),
        shuffle=False,
        num_workers=training_params.get('num_workers', 4),
        pin_memory=True
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='gvae-{epoch:02d}-{train_loss:.2f}',  # Changed from val_loss to train_loss
        monitor='train_loss',  # Changed from val_loss to train_loss
        mode='min',
        save_top_k=3
    )

    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="gvae")

    # Find the latest checkpoint
    checkpoint_dir = 'checkpoints'
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)


    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_params.get('max_epochs', 100),
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=training_params.get('gradient_clip_val', 1.0),
        log_every_n_steps=10
    )

    # Train model
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=latest_checkpoint  # Use ckpt_path instead of resume_from_checkpoint

    )

    return lightning_model, trainer

    # Example usage with variable-sized graphs
class VariableSizeGraphDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


# Analysis function to check graph size distribution
def analyze_graph_sizes(dataset):
    sizes = []
    for data in dataset:
        sizes.append(data.num_nodes)
    return {
        'min_nodes': min(sizes),
        'max_nodes': max(sizes),
        'mean_nodes': sum(sizes) / len(sizes),
        'std_nodes': torch.tensor(sizes).std().item()
    }


if __name__ == "__main__":
    # Example of handling different sized graphs
    model_params = {
        'node_features': 27,
        'hidden_dim': 64,
        'latent_dim': 32,
        'max_nodes': 100  # This is now just an upper bound
    }

    training_params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 100,
        'num_workers': 4,
        'gradient_clip_val': 1.0,
        'beta': 0.1
    }


    # Example of generating molecules of different sizes
    def generate_diverse_molecules(model, size_range=(20, 50), num_molecules=10):
        model.eval()
        molecules = []
        with torch.no_grad():
            for _ in range(num_molecules):
                # Randomly sample a size within the range
                num_nodes = torch.randint(size_range[0], size_range[1], (1,)).item()
                mol = model.generate(num_nodes=num_nodes)
                molecules.append(mol)
        return molecules
