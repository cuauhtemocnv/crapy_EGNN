from models/utils import *
from models/egnn import *
import torch
from torch import nn

# Hyperparameters
num_nodes = 6
feature_dim = 1
hidden_dim =10
output_dim = 1
cutoff = 1.5
device = 'cpu'
n_samples=10
# Example: Prune weights, freeze pruned parameters, and train on uncertain data
# Example model instantiation and optimizer setup
model = EGNN(
    in_node_nf=feature_dim,
    hidden_nf=hidden_dim,
    out_node_nf=output_dim,
    n_layers=3
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
k=0
j=0
model.train()  # Set model to training mode
loss_fn = nn.MSELoss()

while j<100:
    train_data = generate_data(num_nodes, feature_dim, cutoff)
    # Uncertainty sampling for training
    node_features, node_coords, edge_indices = train_data  # Assuming train_data is in this format
    target_coords=update_coordinates(node_coords, node_features)
    target_h=torch.norm(node_features)
    # Get mean predictions and uncertainties
    (h_mean, h_uncertainty), (x_mean, x_uncertainty) = model.forward_with_uncertainty(node_features, node_coords, edge_indices, n_samples)
    # Use uncertainty to filter out high-uncertainty predictions for training
    mask = (h_uncertainty > 0.0002) & (x_uncertainty > 0.6)
    # mask = (x_uncertainty > 0.5)
    # print("uncer hx",h_uncertainty.item(),x_uncertainty.item())
    if (mask and j<100):
        # Perform pruning
        prune_weights(model)
        # Freeze pruned parameters
        freeze_parameters(model)
        # Apply the mask to h_mean and x_mean (you need to ensure that the mask applies element-wise)
        optimizer.zero_grad()
        loss_coord = loss_fn(target_coords,x_mean)  # Compute loss on uncertain data
        loss_h= loss_fn(target_h,torch.norm(h_mean))
        loss= loss_coord+0.0*loss_h
        loss.backward()
        optimizer.step()
        print("Got example",loss.item(),"looshx", loss_h.item(),loss_coord.item(),"uncertx",x_uncertainty,j,k)
        j+=1
    if ((k<100) and (mask==False)):
        optimizer.zero_grad()
        loss_coord = loss_fn(target_coords,x_mean)  # Compute loss on uncertain data
        loss_h= loss_fn(target_h,torch.norm(h_mean))
        loss= loss_coord+0.00*loss_h
        loss.backward()
        optimizer.step()
        print("loss",loss.item(),k)
        k+=1


#now train a second model to compare

# Hyperparameters
num_nodes = 6
feature_dim = 1
hidden_dim =10
output_dim = 1
cutoff = 1.5
device = 'cpu'
n_samples=10
# Example: Prune weights, freeze pruned parameters, and train on uncertain data
# Example model instantiation and optimizer setup
egnn2 = EGNN(
    in_node_nf=feature_dim,
    hidden_nf=hidden_dim,
    out_node_nf=output_dim,
    n_layers=3
).to(device)
optimizer = torch.optim.Adam(egnn2.parameters(), lr=1e-3)
k=0
j=0
egnn2.train()  # Set model to training mode
loss_fn = nn.MSELoss()

while j<200:
    train_data = generate_data(num_nodes, feature_dim, cutoff)
    # Perform pruning
    # prune_weights(egnn2)
    # # Freeze pruned parameters
    # freeze_parameters(egnn2)
    # Uncertainty sampling for training
    node_features, node_coords, edge_indices = train_data  # Assuming train_data is in this format
    target_coords=update_coordinates(node_coords, node_features)
    target_h=torch.norm(node_features)
    # Get mean predictions and uncertainties
    (h_mean, h_uncertainty), (x_mean, x_uncertainty) = egnn2.forward_with_uncertainty(node_features, node_coords, edge_indices, n_samples)
    # Use uncertainty to filter out high-uncertainty predictions for training
    mask = (h_uncertainty < 0.03) & (x_uncertainty < 0.4)
    mask = (x_uncertainty < 0.6)
    mask = True
    # print("uncer hx",h_uncertainty.item(),x_uncertainty.item())
    if mask:
        # Apply the mask to h_mean and x_mean (you need to ensure that the mask applies element-wise)
        optimizer.zero_grad()
        loss_coord = loss_fn(target_coords,x_mean)  # Compute loss on uncertain data
        loss_h= loss_fn(target_h,torch.norm(h_mean))
        loss= loss_coord+0.0*loss_h
        loss.backward()
        optimizer.step()
        print("Got example",loss.item(),"looshx", loss_h.item(),loss_coord.item(),"uncertx",x_uncertainty,j)
        j+=1
        k+=1
    # if k<100:
    #     optimizer.zero_grad()
    #     loss_coord = loss_fn(target_coords,x_mean)  # Compute loss on uncertain data
    #     loss_h= loss_fn(target_h,torch.norm(h_mean))
    #     loss= loss_coord+0.00*loss_h
    #     loss.backward()
    #     optimizer.step()
    #     print("loss",loss.item(),k)
    #     k+=1


#plot the result
egnn2.eval()
model.eval()
n_samples=10
num_nodes = 12
loss_fn = nn.MSELoss()
for _ in range(10):
    train_data = generate_data(num_nodes, feature_dim, cutoff)
    node_features, node_coords, edge_indices = train_data  # Assuming train_data is in this format
    target_coords=update_coordinates(node_coords, node_features)
    target_h=torch.norm(node_features)
    (h_mean, h_uncertainty), (x_mean, x_uncertainty) = model.forward_with_uncertainty(node_features, node_coords,edge_indices)
    loss_coordm = loss_fn(target_coords,x_mean)
    (h_mean, h_uncertainty), (x_mean2, x_uncertainty2) = egnn2.forward_with_uncertainty(node_features, node_coords, edge_indices)
    loss_coordegnn2 = loss_fn(target_coords,x_mean2)
    print((x_uncertainty-x_uncertainty2).item())
    print(loss_coordm.item()-loss_coordegnn2.item())
    plt.figure(figsize=(6, 6))
    # plt.scatter(node_coords[:, 0], node_coords[:, 1], color='blue', label='Original Coordinates')
    plt.scatter(x_mean.detach().numpy()[:, 0], x_mean.detach().numpy()[:, 1], color='red', label='Predicted Coordinates1')
    plt.scatter(x_mean2.detach().numpy()[:, 0], x_mean2.detach().numpy()[:, 1], color='yellow', label='Predicted Coordinates2')
    plt.scatter(target_coords[:, 0], target_coords[:, 1], color='brown', label='targeT Coordinates')
    # plt.scatter(dif[:, 0], dif[:, 1], color='yellow', label='diff')
    plt.legend()
    plt.title("high hx")
    plt.show()


#now lets see how the models perform with crap data
egnn.eval()
n_samples=10
num_nodes = 6
loss_fn = nn.MSELoss()
for _ in range(1000):
    train_data = generate_data(num_nodes, feature_dim, cutoff)
    node_features, node_coords, edge_indices = train_data  # Assuming train_data is in this format
    target_coords=update_coordinates(node_coords, node_features)
    target_h=torch.norm(node_features)
    # Get mean predictions and uncertainties
    (h_mean, h_uncertainty), (x_mean, x_uncertainty) = egnn.forward_with_uncertainty(node_features, node_coords, edge_indices, n_samples)
    loss_coord = loss_fn(target_coords,x_mean)  # Compute loss on uncertain data

    # Use uncertainty to filter out high-uncertainty predictions for training
    mask = (x_uncertainty > 0.7)
    if mask:
        print(loss_coord.item(),x_uncertainty.item())
        # Visualize coordinates before and after training
        plt.figure(figsize=(6, 6))
        plt.scatter(node_coords[:, 0], node_coords[:, 1], color='blue', label='Original Coordinates')
        plt.scatter(x_mean.detach().numpy()[:, 0], x_mean.detach().numpy()[:, 1], color='red', label='Predicted Coordinates')
        plt.scatter(target_coords[:, 0], target_coords[:, 1], color='brown', label='targeT Coordinates')
        # plt.scatter(dif[:, 0], dif[:, 1], color='yellow', label='diff')
        plt.legend()
        plt.title("high hx")
        plt.show()
    mask = (x_uncertainty <0.3)
    if mask:
        print(loss_coord.item(),x_uncertainty.item())
        # Visualize coordinates before and after training
        plt.figure(figsize=(6, 6))
        plt.scatter(node_coords[:, 0], node_coords[:, 1], color='blue', label='Original Coordinates')
        plt.scatter(x_mean.detach().numpy()[:, 0], x_mean.detach().numpy()[:, 1], color='red', label='Predicted Coordinates')
        plt.scatter(target_coords[:, 0], target_coords[:, 1], color='brown', label='targeT Coordinates')
        # plt.scatter(dif[:, 0], dif[:, 1], color='yellow', label='diff')
        plt.legend()
        plt.title("low hx")
        plt.show()
