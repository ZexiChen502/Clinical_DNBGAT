#%%
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from utils import convert_timepoint_to_label, calculate_correlation
from layer import GAT
import os
import json
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F


def set_seed(seed=42):
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch CPU operations.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU operations.
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_G_data(
        mirnaem,
        dnb_gene_list,
        clinical,
        label,
        cutoff=0.8,
        transformation_type='none'
):

    # Select the type of transformation based on the parameter
    if transformation_type == 'log':
        transformed_data = np.log1p(mirnaem.loc[:, dnb_gene_list]).values
    elif transformation_type == 'standardize':
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(mirnaem.loc[:, dnb_gene_list])
    elif transformation_type == 'none':
        transformed_data = mirnaem.loc[:, dnb_gene_list].values
    else:
        raise ValueError("Invalid transformation type. Choose 'log', 'standardize', or 'none'.")

    matrix = calculate_correlation(clinical)

    adjacency_matrix = matrix.where(matrix >= cutoff, 0)

    # data
    X = torch.tensor(transformed_data, dtype=torch.float32)

    # edge
    adjacency_matrix_np = adjacency_matrix.values
    edges = np.nonzero(adjacency_matrix_np)
    edges_array = np.array(edges)
    edge_weights = adjacency_matrix_np[edges]
    edge_index = torch.tensor(edges_array, dtype=torch.long).contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

    print("Number of edges:", edges_array.shape[1])

    # label
    labels = torch.tensor(label.values, dtype=torch.long)

    G_data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    return G_data


def create_model(num_node_features, num_classes, hidden_dims, heads, dropout=0.1):
    model = GAT(num_node_features=num_node_features, num_classes=num_classes, hidden_dims=hidden_dims, heads=heads, dropout=dropout)
    return model


def mask_node(
        data,
        train_index,
        test_index,
        proportion4train=0.8,
):

    num_nodes = data.x.shape[0]

    train_nodes_len = len(train_index)
    train_size = int(train_nodes_len * proportion4train)

    np.random.shuffle(train_index)
    train_indices = train_index[:train_size]
    val_indices = train_index[train_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_index] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    node_role = np.full(num_nodes, 'Test', dtype='U5')
    node_role[train_indices] = 'Train'
    node_role[val_indices] = 'Val'
    data.node_role = node_role

    return data


def train(
        mirnaem,
        dnb_gene_list,
        clinical,
        label,
        train_index,
        test_index,
        cutoff=0.8,
        proportion4train=0.8,
        hidden_dims=None,
        lr=0.01,
        epochs=400,
        seed_value=42,
        heads=4,
        dropout=0.1,
        transformation_type='none'
):
    if hidden_dims is None:
        hidden_dims = [16]
    set_seed(seed_value)

    data = create_G_data(mirnaem, dnb_gene_list, clinical, label, cutoff, transformation_type=transformation_type)
    data = mask_node(data, train_index=train_index, test_index=test_index, proportion4train=proportion4train)

    model = create_model(data.x.shape[1], 3, hidden_dims, heads=heads, dropout=dropout)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        _, pred = out.max(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}, '
              f'Train Acc: {train_acc}, Val Acc: {val_acc}')

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "LOSS.png"), dpi=300)
    plt.show()

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    # plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "ACC.png"), dpi=300)
    plt.show()

    # Save results and model
    result_df = pd.DataFrame({
        'Predicted Class': pred.numpy(),
        'True Class': data.y.numpy(),
        'Node Role': data.node_role
    })
    result_df.to_csv(os.path.join(save_dir, "result.csv"))

    test_df = result_df[result_df['Node Role'] == 'Test']
    accuracy = accuracy_score(test_df['True Class'], test_df['Predicted Class'])
    print(f"The accuracy of the test set is: {accuracy:.3f}")

    cm = confusion_matrix(test_df['True Class'], test_df['Predicted Class'])
    # 可视化
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk', 'Cancer'],
                yticklabels=['Low Risk', 'High Risk', 'Cancer'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Test Set')
    plt.savefig(os.path.join(save_dir, "Confusion_Matrix.png"), dpi=300)
    plt.show()

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    model_params = {
        'num_node_features': data.x.size(1),
        'num_classes': 3,
        'hidden_dims': hidden_dims
    }
    with open(os.path.join(save_dir, 'model_params.json'), 'w') as file:
        json.dump(model_params, file)

    # ROC
    model.eval()
    out = model(data)
    probs = F.softmax(out, dim=1)
    test_probs = probs[data.test_mask]  # 仅选择测试数据的输出概率

    test_probs_class2 = test_probs[:, 2].detach().numpy()
    test_labels_modified = (data.y[data.test_mask] == 2).long()

    fpr, tpr, _ = roc_curve(test_labels_modified, test_probs_class2)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Cancer')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "ROC_Class2.png"), dpi=300)
    plt.show()
    plt.close()

    test_probs_class1 = test_probs[:, 1].detach().numpy()
    test_labels_modified = (data.y[data.test_mask] == 1).long()
    fpr, tpr, _ = roc_curve(test_labels_modified, test_probs_class1)
    roc_auc = auc(fpr, tpr)


    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for High Risk')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "ROC_Class1.png"), dpi=300)
    plt.show()
    plt.close()

    test_probs_class0 = test_probs[:, 0].detach().numpy()
    test_labels_modified = (data.y[data.test_mask] == 0).long()
    fpr, tpr, _ = roc_curve(test_labels_modified, test_probs_class0)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Low Risk')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "ROC_Class0.png"), dpi=300)
    plt.show()
    plt.close()

    return result_df, model



