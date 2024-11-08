So this contains a lot of test notebooks and playing around. Currently active are the Graph_VAE_protein notebook.

To use there is a requirements file to install any required libs, use the prep.py to copy the smallest number of files you are looking to use.
I copied the smallest 5000 files into a directory which i used for basic testing.I also had a 100 smallest just to check dimensions of the vae were working etc.

There is then a utils file, a proteinanalyzer file which generates dfs for the relevant pdbs and a graphcreatoronehotencoder.
The graphcreatoronehotencoder creates a graph area from the dfs created by the proteinanalyzer.

There are helper  methods for ploting and printing the graphs,
I will create as well 3d plots in the protein analyzer or graphcreator...

I uploaded as well a plotly 3d in protein_notebook2, just as an example...i will create the plotting as helper functions as mentioned some place more appropriate

Feel free to add new stuff...i will do a deeper clean of the whole repo some time soon!!


Current ideas are to try and plot some stuff in the latent space. If we knew the families then they should be visible in the latent space if the vae is working correctly...
Visualizing the latent space of a Variational Graph Autoencoder (VGAE) trained on protein data is an excellent way to understand how well the model captures structural and functional similarities in the data. Here are some common methods and steps to visualize the latent space, focusing on approaches that are especially helpful for protein data:

1. Extract Latent Embeddings
After training your VGAE, extract the latent embeddings from the encoder for each protein graph or each node within a protein graph, depending on whether you're visualizing protein structures as a whole or specific atomic/local features.
python
Copy code
# Assuming your VGAE model is trained and `x` and `edge_index` represent your graph data
z = model.vgae.encode(x, edge_index)  # z now contains the latent embeddings
2. Dimensionality Reduction Techniques
Latent spaces in VGAEs are often high-dimensional, so applying dimensionality reduction can help visualize the data in 2D or 3D. Common techniques are:
t-SNE (t-Distributed Stochastic Neighbor Embedding): Effective for high-dimensional data and commonly used in embedding visualization. However, it’s computationally intensive and doesn’t always preserve global structure.
UMAP (Uniform Manifold Approximation and Projection): Often better at preserving both local and global structure and is faster than t-SNE.
PCA (Principal Component Analysis): Simple and fast but may not capture complex non-linear structures as well.
python
Copy code
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply t-SNE to reduce dimensionality to 2D
z_2d = TSNE(n_components=2).fit_transform(z.detach().numpy())

# Plotting
plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D Visualization of Latent Space')
plt.show()
For UMAP:
python
Copy code
from umap import UMAP
z_2d = UMAP(n_components=2).fit_transform(z.detach().numpy())
You can also apply 3D t-SNE or UMAP (set n_components=3), then use libraries like Plotly or matplotlib for interactive 3D plots.

3. Coloring by Labels or Biological Features
To understand what the latent space represents, color-code the points based on relevant biological features. For proteins, useful labels might include:
Protein family or domain: If your proteins belong to different families, label each family with a unique color.
Function or active site labels: If you have information on the functional roles of each protein or node (e.g., binding sites, catalytic residues), use these labels.
Structural similarity or secondary structure: Label nodes or embeddings based on secondary structures, such as alpha helices, beta sheets, and loops.
python
Copy code
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=protein_labels, cmap='viridis', alpha=0.7)
plt.colorbar(label='Protein Family')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Latent Space Colored by Protein Family')
plt.show()
4. Visualizing the Progression of Training
Another interesting visualization is to observe how the latent space changes over the course of training. You can save embeddings at different epochs and visualize them together to see how the structure of the latent space emerges and improves.
python
Copy code
z_epoch1 = model.vgae.encode(x, edge_index).detach().numpy()  # Latent space after 1st epoch
z_epoch_end = model.vgae.encode(x, edge_index).detach().numpy()  # Latent space at final epoch

# Plot comparison (assuming t-SNE or UMAP on both z_epoch1 and z_epoch_end)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(z_epoch1[:, 0], z_epoch1[:, 1], c=protein_labels, cmap='viridis', alpha=0.6)
ax1.set_title('Epoch 1')
ax2.scatter(z_epoch_end[:, 0], z_epoch_end[:, 1], c=protein_labels, cmap='viridis', alpha=0.6)
ax2.set_title('Final Epoch')
plt.show()
5. Pairing with Protein Structure Visualizations
To get a more intuitive sense of how the latent representations relate to physical structures, you can use structure visualization libraries (e.g., PyMOL, Py3Dmol, or nglview) to display representative protein structures. For instance:
Select clusters from the latent space visualization.
Map these back to the original protein structures.
Display selected representative structures in 3D to see if similar embeddings correlate with similar 3D shapes or binding sites.
6. Latent Space Density Visualization
Density plots can reveal regions of high and low density within the latent space, which may correspond to common or rare protein structures.
Kernel density estimation (KDE) or heatmaps on the latent embeddings can provide this kind of insight.
python
Copy code
import seaborn as sns
sns.kdeplot(x=z_2d[:, 0], y=z_2d[:, 1], cmap="Blues", shade=True, bw_adjust=0.5)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Density Plot of Latent Space')
plt.show()
7. Interpretation and Biological Insight
After visualizing, interpret the clusters or patterns observed:
Clustering: Similar proteins (in terms of function or structure) should ideally be grouped closely in the latent space if the VGAE is effective.
Outliers: Outlier embeddings can indicate proteins or regions that do not fit typical structures, which may be interesting for further analysis.


Using the number of nearby atoms as a criterion for defining edges in your protein graph is a great approach, especially because it allows you to create edges that capture local density and potentially represent interactions better than just a fixed distance threshold. Here are a few methods for incorporating this information effectively:

1. K-Nearest Neighbors (K-NN) Graph Construction
For each atom (node), create edges to its k-nearest neighbors (where k is the number of nearby atoms you want to consider). This approach guarantees that each node will have exactly k neighbors, allowing you to capture a local neighborhood with consistent connectivity.
Dynamic K Selection: If the nearby atom count varies significantly across atoms, you could set a local k value per atom based on its surrounding density (e.g., higher k for atoms in denser regions).
2. Radius-based K-Nearest Neighbors (K-RNN)
If you have a desired number of neighbors (say, around 6-8 atoms), but the distribution is variable, you can combine a radius and K-NN approach. Set a fixed radius to look within (e.g., 5 Å) but cap the maximum neighbors per node to the nearest k atoms within that radius.
This approach ensures that nodes maintain local density without overly dense connections in high-density areas.
3. Adaptive Edge Creation Based on Local Density
Use a density-based approach where each node has edges based on the local density of atoms. For example:
Define a "nearby atom threshold," say 6-8 atoms within a fixed radius.
For each node, dynamically adjust the radius to achieve the target neighbor count. This will make edges denser in high-density regions and sparser in low-density regions.
Alternatively, you could make a weighted edge based on density. For instance, nodes in denser regions have stronger or closer edges than nodes in sparser regions.
4. Weighted Edge Based on Distance Ranking
Instead of treating each neighbor equally, weight each edge based on its rank within the k-nearest atoms. Closer atoms would receive higher weights, and farther neighbors would get lower weights.
This weighting can be achieved by defining edge weights as the inverse distance rank (e.g., 1/rank). The model can learn better relationships by prioritizing closer atoms.
5. Graph Pruning with Minimum Spanning Trees
After defining initial edges based on the number of nearby atoms, use a minimum spanning tree (MST) algorithm to prune edges selectively.
This will ensure each node is connected within its local neighborhood, while reducing the total number of edges and preserving important connectivity patterns.
6. Edge Embedding with Neighbor Count Feature
Use the neighbor count as a feature on each edge. For each edge, you could add the count of other atoms within the vicinity of the connected nodes as an edge feature.
For example, if an atom A has 5 neighbors and atom B has 7 neighbors, the edge feature between A and B could be [5, 7]. This feature could help your model learn the density or locality around each edge connection.
7. Contrastive Edge Learning Based on Density
You could apply a contrastive learning approach by defining an auxiliary task that learns to distinguish between "close neighbor edges" and "non-close neighbor edges." For instance, create edges based on the number of nearby atoms and label those as positive edges, while other longer-distance edges can serve as negatives. This can help your model prioritize learning from meaningful edges.


For the edges 
Improving edge learning in protein-related graphs can be challenging, especially in 3D molecular structures, where edges often represent critical biochemical interactions. Since you’re working with a protein graph and are already using a distance metric with 3D (xyz) features, here are a few strategies you could try:

1. Enhanced Edge Feature Engineering
Angular and Dihedral Features: Beyond distance and xyz coordinates, consider adding angular and dihedral angles between connected nodes. These features capture the relative orientation of atoms or residues and are especially useful in proteins, where spatial orientation heavily influences function.
Edge Attributes: Incorporate edge-specific attributes such as bond type (e.g., covalent, hydrogen, or hydrophobic interactions) if your model is aware of them. For protein-protein interaction networks or similar complex graphs, incorporating physicochemical properties (e.g., hydrophobicity) can also help the model differentiate types of edges.
2. Edge-Weighted Graph Networks
Use a weighted adjacency matrix where each edge weight reflects a function of the distance metric or other biochemical properties. This can help the model emphasize shorter or stronger interactions and avoid overemphasizing long-distance, weaker connections.
3. Improved Distance Representation
Radial Basis Function (RBF): Transform the distance metric using an RBF or Gaussian kernel, which can help the model learn continuous representations of distances. RBF kernels map the distance into a continuous feature space, which often aids learning in 3D environments.
Distance Binning: Quantize distances into bins so that different distance intervals (e.g., <4 Å, 4-8 Å, etc.) are treated differently, allowing the model to capture distinctions in short-range versus long-range interactions.
4. Attention Mechanisms for Edges
Edge-Conditioned Convolutions (ECC): In these architectures, convolution weights are conditioned on the edge features, enabling the model to learn different weights for different types of interactions. ECC can help capture unique characteristics of edges beyond just the spatial features.
Graph Attention Networks (GAT): Integrate a GAT or edge attention mechanism that explicitly weighs each edge based on both node and edge features. This can guide the model to pay more attention to critical edges that represent significant interactions.
5. Data Augmentation for Edge Learning
Geometric Augmentation: Apply rotations, translations, or small perturbations to your graph's 3D coordinates. This can help the model generalize better and learn spatial dependencies more effectively, especially for edges defined by spatial relationships.
Edge Masking: Randomly mask or drop edges during training to encourage the model to learn a robust representation of edges. This can prevent the model from relying too much on certain edges and help it generalize better.
6. Loss Function Adjustments
Edge-Specific Loss: If the model is not learning edges well, consider adding an edge-specific loss term. For example, you could penalize errors on edge representations more heavily if they are within critical distance thresholds. You might also use a regularization term to emphasize important edges.
Contrastive Loss on Edge Embeddings: If you have access to labels or classifications of edges (e.g., specific types of interactions), a contrastive or triplet loss could be useful for pulling similar types of edges closer in the embedding space while pushing dissimilar ones apart.
7. Layer-wise Optimization for Edges
If you’re using multiple layers in your GNN, try increasing the number of edge convolution layers. Each additional layer allows the network to capture a larger neighborhood in the graph, which could improve edge feature learning, especially for longer-range dependencies in protein graphs.
