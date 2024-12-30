import torch.nn as nn
import torch.nn.init as init
import torch
def generate_data(num_nodes, feature_dim, cutoff=0.1):
    node_features = torch.rand((num_nodes, feature_dim))  # Random node features
    node_coords = torch.rand((num_nodes, 2))  # Random 2D coordinates
    edge_index, _ = get_edges(node_coords, cutoff)  # Generate edges
    return node_features, node_coords, edge_index

def update_coordinates(node_coords, masses, alpha=1.0):
    # Ensure masses is of shape [20, 1] for broadcasting
    masses = masses.squeeze()  # Shape: [20]
    
    # Compute centroid using mass-weighted average of coordinates
    mass_sum = masses.sum()
    centroid = (node_coords.T * masses).T.sum(dim=0) / mass_sum  # Shape: [2]
    
    # Move nodes towards centroid proportionally to their masses
    displacement = alpha * masses[:, None] * (centroid - node_coords)  # Shape: [20, 2]
    
    # Update coordinates
    updated_coords = node_coords + displacement
    
    return updated_coords

def prune_weights(model, threshold=1e-6):
    for param in model.parameters():
        param.data = torch.where(torch.abs(param) < threshold, torch.tensor(0.0, device=param.device), param.data)
def freeze_parameters(model, threshold=1e-6):
    for name, param in model.named_parameters():
        # Freeze pruned parameters (those that were set to zero)
        if torch.all(torch.abs(param.data) < threshold):
            param.requires_grad = False
        else:
            param.requires_grad = True
def get_edges(points, cutoff):
    """
    Creates edge index and edge distances for a point cloud within a cutoff radius.

    Args:
        points: A PyTorch tensor of shape [num_points, 3] representing the point cloud.
        cutoff: The cutoff radius.

    Returns:
        edge_index: A PyTorch tensor of shape [2, num_edges] representing the edge index.
        edge_attr: A PyTorch tensor of shape [num_edges, 1] representing the edge distances.
    """

    dist_mat = torch.cdist(points, points)
    mask = dist_mat > cutoff
    dist_mat[mask] = 0

    edge_index = torch.nonzero(dist_mat, as_tuple=True)
    edge_attr = dist_mat[edge_index[0], edge_index[1]].unsqueeze(1)

    return edge_index, edge_attr    
