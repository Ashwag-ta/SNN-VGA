# General Imports
import pandas as pd
import numpy as np
import sys
import copy
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
import random as rd
from tabulate import tabulate

# Pytorch
import torch

# Pytorch Geometric
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import NormalizeFeatures

#Sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, roc_curve, average_precision_score, RocCurveDisplay

# Our Method
from SNNVGA_model import LinkPred



# Training function
def train(model, data, optimizer, args):
    KL_divergence = 0
    data.to(args["device"])
    model.train()
    optimizer.zero_grad()
    mu, logvar, z, x_cons = model(data)
    reconstruction_loss = model.loss_BCE( x_cons , data.edge_label)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim= 1)
    total_loss = reconstruction_loss + KL_divergence
    total_loss.backward()
    optimizer.step()
    
    return total_loss



# Testing function
@torch.no_grad()
def test(model, data, args ):
    model.eval()
    data.to(args["device"])
    mu, logvar, z, x_cons = model(data)
    y_pred_binary = (x_cons > 0.5).float()
    
    # Calculate evaluation metrics
    auc = roc_auc_score(data.edge_label.cpu().numpy(), x_cons.cpu().numpy())
    precision = precision_score(data.edge_label.cpu().numpy(), y_pred_binary.cpu().numpy() , zero_division=1)
    accuracy = accuracy_score(data.edge_label.cpu().numpy(), y_pred_binary.cpu().numpy())
    recall = recall_score(data.edge_label.cpu().numpy(), y_pred_binary.cpu().numpy())
    f1 = f1_score(data.edge_label.cpu().numpy(), y_pred_binary.cpu().numpy())
    average_precision = average_precision_score(data.edge_label.cpu().numpy(), x_cons.cpu().numpy())
    y_true= data.edge_label.cpu().numpy()
    y_pred= x_cons.cpu().numpy()
    
    return auc , accuracy , precision , recall , f1 , average_precision , y_true , y_pred



def main():
    # Hyperparameters and settings
    args = {
        "device" : 'cuda' if torch.cuda.is_available() else 'cpu',
        "hidden_dim" : 128,
        "out_dim" : 64,
        "epochs" : 50,
        "seed" : 1171,
        "RR" : sys.argv[1]
        }

    # Initialization
    final_val_auc = final_val_acc = final_val_prec = final_val_recall = final_val_f1 = 0
    final_test_auc = final_test_acc = final_test_prec = final_test_recall = final_test_f1 = 0
    bestmode_mean_table = []
    val_auc_list, val_accuracy_list, val_precision_list, val_recall_list, val_f1_list, val_ap_list = [], [], [], [], [] , []
    bestmodel_val_auc_list, bestmodel_val_accuracy_list, bestmodel_val_precision_list, bestmodel_val_recall_list, bestmodel_val_f1_list, bestmodel_val_ap_list = [], [], [], [], [] , []
    bestmodel_test_auc_list, bestmodel_test_accuracy_list, bestmodel_test_precision_list, bestmodel_test_recall_list, bestmodel_test_f1_list, bestmodel_test_ap_list  = [], [], [], [], [] , []

    # Initialize lists to store mean fpr, tpr, and auc for different folds to plot ROC
    mean_fpr_test = np.linspace(0, 1, 100)
    tprs_test = []
    aucs_test = []
    fig_test, ax_test = plt.subplots(figsize=(6, 6))

    # Prepare cross-validation settings
    outer_folds = 10
    inner_folds = 5
    outer_skf = KFold(n_splits=outer_folds, shuffle=True, random_state = args["seed"])
    inner_skf = KFold(n_splits=inner_folds, shuffle=True, random_state = args["seed"])

    # Set seeds for reproducibility
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    rd.seed(args["seed"])

    # Read input features
    list_features_df = pd.read_csv('Disease_features.csv' , header=None)

    # Load edge indices and labels based on RR value
    RR = args["RR"]
    if RR == "0" :
        edge_index_df = pd.read_csv('DDI_RR0_positive_edges.csv') 
        edge_labels_df = pd.read_csv('DDI_RR0_positive_labels.csv') 
    elif RR == "1":    
        edge_index_df = pd.read_csv('DDI_RR1_positive_edges.csv')
        edge_labels_df = pd.read_csv('DDI_RR1_positive_labels.csv')
    else:
        raise ValueError("RR must be either '0' or '1'")

    # Prepare data
    x = torch.tensor(list_features_df.values, dtype=torch.float, device = args["device"])
    edge_index = torch.tensor(edge_index_df[['source', 'target']].values, dtype=torch.long, device = args["device"]).t().contiguous()
    edge_index = edge_index[:, torch.randperm(edge_index.size(1))]
    edge_label = torch.tensor(edge_labels_df['label'].values, dtype=torch.float32, device = args["device"])
    data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
    transform = NormalizeFeatures()
    data = transform(data)
    in_dim = data.num_features  # Input dimension

    # Generate negative samples
    num_neg_samples = int (data.edge_label.size(0))
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
    num_neg_samples=num_neg_samples, method='sparse')

    edge_index_transposed = edge_index.t()
    edge_label_transposed = edge_label.t()


    # Start outer cross-validation
    for fold, (train_ids, test_ids) in enumerate(outer_skf.split(edge_index_transposed, edge_label_transposed)):
        model = LinkPred(in_dim, args["hidden_dim"], args["out_dim"]).to(args["device"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        best_model = model
        max_val_auc = 0
        best_av_model_per = 0

        # Prepare training and validation data
        train_edge_index_source = edge_index_transposed[torch.tensor(train_ids)][:, 0]
        train_edge_index_target = edge_index_transposed[torch.tensor(train_ids)][:, 1]
        train_edge_index = torch.stack((train_edge_index_source, train_edge_index_target), dim=1)
        train_edge_labels = edge_label_transposed[train_ids]

        # Prepare test data
        test_edge_index_source = edge_index_transposed[torch.tensor(test_ids)][:, 0]
        test_edge_index_target = edge_index_transposed[torch.tensor(test_ids)][:, 1]
        pos_test_edge_label_index = torch.stack((test_edge_index_source, test_edge_index_target))
        pos_test_edge_label = edge_label_transposed[test_ids]
        test_edge_index = torch.stack((train_edge_index_source, train_edge_index_target))
        test_data_x = data.x
        test_data_pos = Data(x=test_data_x, edge_index=test_edge_index ,edge_label_index= pos_test_edge_label_index , edge_label=pos_test_edge_label )

        # Split negative samples for training, validation, and testing
        num_neg_samples_train =  train_edge_labels.size()[0]
        num_neg_samples_test = test_data_pos.edge_label.size(0)
        neg_train_edge_label_index = neg_edge_index[:, :num_neg_samples_train]
    
        # Prepare negative samples for test set
        neg_test_edge_label_index = neg_edge_index[:, num_neg_samples_train:num_neg_samples_train+num_neg_samples_test]
        neg_edge_label_test = (test_data_pos.edge_label.new_zeros(neg_test_edge_label_index.size(1)))
        test_data_neg = Data(x=test_data_x, edge_label_index=neg_test_edge_label_index, edge_label=neg_edge_label_test )
        # Concatenate positive and negative edges for the test data
        curr_test_data = test_data_pos.clone()  
        curr_test_data.edge_label_index = torch.cat([test_data_pos.edge_label_index, test_data_neg.edge_label_index], dim=1)
        curr_test_data.edge_label = torch.cat([test_data_pos.edge_label, test_data_neg.edge_label])

        # Inner cross-validation
        for inner_fold, (inner_train_ids, val_ids) in enumerate(inner_skf.split(train_edge_index, train_edge_labels)):
            
            # Prepare training set
            inner_train_edge_index_source = train_edge_index[torch.tensor(inner_train_ids)][:, 0]
            inner_train_edge_index_target = train_edge_index[torch.tensor(inner_train_ids)][:, 1]
            pos_inner_train_edge_label_index = torch.stack((inner_train_edge_index_source, inner_train_edge_index_target))
            pos_inner_train_edge_label = train_edge_labels[inner_train_ids]
            train_data_pos = Data(x=data.x, edge_index=pos_inner_train_edge_label_index , edge_label_index=pos_inner_train_edge_label_index, edge_label=pos_inner_train_edge_label)

            # Prepare negative samples for training set
            num_neg_samples_inner_train = train_data_pos.edge_label.size(0)
            neg_inner_train_edge_label_index = neg_train_edge_label_index[:, :num_neg_samples_inner_train]
            neg_inner_train_edge_label =  train_data_pos.edge_label.new_zeros(neg_inner_train_edge_label_index .size(1))
            train_data_neg = Data(x=data.x,  edge_label_index=neg_inner_train_edge_label_index, edge_label=neg_inner_train_edge_label )
            # Concatenate positive and negative edges for the training data
            curr_inner_train_data = train_data_pos.clone()
            curr_inner_train_data.edge_label_index = torch.cat([train_data_pos.edge_label_index, train_data_neg.edge_label_index], dim=1)
            curr_inner_train_data.edge_label = torch.cat([train_data_pos.edge_label, train_data_neg.edge_label])

            # Prepare validation set
            val_edge_index_source = train_edge_index[torch.tensor(val_ids)][:, 0]
            val_edge_index_target = train_edge_index[torch.tensor(val_ids)][:, 1]
            pos_val_edge_label_index = torch.stack((val_edge_index_source, val_edge_index_target))
            pos_val_edge_label = train_edge_labels[val_ids]
            pos_val_edge_index = train_data_pos.edge_index
            val_data_x= data.x
            val_data_pos = Data(x=val_data_x, edge_index=pos_val_edge_index , edge_label_index= pos_val_edge_label_index , edge_label=pos_val_edge_label)

            # Prepare negative samples for validation set
            num_neg_samples_val = val_data_pos.edge_label.size(0)
            neg_val_edge_label_index = neg_train_edge_label_index[:, num_neg_samples_inner_train:num_neg_samples_inner_train+num_neg_samples_val]
            neg_edge_label_val = (val_data_pos.edge_label.new_zeros(neg_val_edge_label_index.size(1)))
            val_data_neg = Data(x=val_data_x, edge_label_index=neg_val_edge_label_index, edge_label=neg_edge_label_val )
            # Concatenate positive and negative edges for the validation data
            curr_val_data = val_data_pos.clone()  
            curr_val_data.edge_label_index = torch.cat([val_data_pos.edge_label_index, val_data_neg.edge_label_index], dim=1)
            curr_val_data.edge_label = torch.cat([val_data_pos.edge_label, val_data_neg.edge_label])

            # Train the model
            for epoch in range(1, args["epochs"]):
              loss = train(model, curr_inner_train_data, optimizer, args)
              log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {:.5f}'
              val_auc, val_accuracy, val_precision, val_recall, val_f1, val_average_precision, y_val, pred_val = test(model, curr_val_data, args)

              # Store metrics for each iteration
              val_auc_list.append(val_auc)
              val_accuracy_list.append(val_accuracy)
              val_precision_list.append(val_precision)
              val_recall_list.append(val_recall)
              val_f1_list.append(val_f1)
              val_ap_list.append(val_average_precision)
          
              # Save the best model
              if max_val_auc < val_auc:
                 max_val_auc = val_auc
                 best_model = copy.deepcopy(model)

     
        # Test the model 
        Best_val_auc, Best_val_accuracy, Best_val_precision, Best_val_recall, Best_val_f1 , Best_val_average_precision, y_val , pred_val  = test(best_model, curr_val_data, args)
        Best_test_auc, Best_test_accuracy, Best_test_precision, Best_test_recall, Best_test_f1, Best_test_average_precision, y_test, pred_test  = test(best_model, curr_test_data , args)

        # Store the results for the validation and test sets
        bestmodel_val_auc_list.append(Best_val_auc)
        bestmodel_val_accuracy_list.append(Best_val_accuracy)
        bestmodel_val_precision_list.append(Best_val_precision)
        bestmodel_val_recall_list.append(Best_val_recall)
        bestmodel_val_f1_list.append(Best_val_f1)
        bestmodel_val_ap_list.append(Best_val_average_precision)
        bestmodel_test_auc_list.append(Best_test_auc)
        bestmodel_test_accuracy_list.append(Best_test_accuracy)
        bestmodel_test_precision_list.append(Best_test_precision)
        bestmodel_test_recall_list.append(Best_test_recall)
        bestmodel_test_f1_list.append(Best_test_f1)
        bestmodel_test_ap_list.append(Best_test_average_precision)
    
        # Compute area under the curve (AUC) for the test set
        viz_test = RocCurveDisplay.from_predictions(y_test, pred_test , name=f"ROC fold {fold}",ax=ax_test)
        interp_tpr_test = np.interp(mean_fpr_test, viz_test.fpr, viz_test.tpr)
        interp_tpr_test[0] = 0.0
        tprs_test.append(interp_tpr_test)
        aucs_test.append(viz_test.roc_auc)

    # Plot ROC for Test set
    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0
    mean_auc_test = auc(mean_fpr_test, mean_tpr_test)
    std_auc_test = np.std(aucs_test)
    ax_test.plot(
        mean_fpr_test,
        mean_tpr_test,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_test, std_auc_test),
        lw=2,
        alpha=0.8,
    )
    std_tpr_test = np.std(tprs_test, axis=0)
    tprs_upper_test = np.minimum(mean_tpr_test + std_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - std_tpr_test, 0)
    ax_test.fill_between(
        mean_fpr_test,
        tprs_lower_test,
        tprs_upper_test,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax_test.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC curves for each fold of the 10-fold cross-validation",
    )
    ax_test.legend(loc="lower right")
    plt.show()

    # Calculate the average metrics for the model results after finishing all outer folds

    # Average metrics for validation set
    Bestmodel_avg_val_auc = np.mean(bestmodel_val_auc_list)
    Bestmodel_avg_val_accuracy = np.mean(bestmodel_val_accuracy_list)
    Bestmodel_avg_val_precision = np.mean(bestmodel_val_precision_list)
    Bestmodel_avg_val_recall = np.mean(bestmodel_val_recall_list)
    Bestmodel_avg_val_f1 = np.mean(bestmodel_val_f1_list)
    Bestmodel_avg_val_ap = np.mean(bestmodel_val_ap_list)

    # Average metrics for test set
    Bestmodel_avg_test_auc = np.mean(bestmodel_test_auc_list)
    std_auc_test_auc = np.std(bestmodel_test_auc_list)
    Bestmodel_avg_test_accuracy = np.mean(bestmodel_test_accuracy_list)
    std_auc_test_accuracy = np.std(bestmodel_test_accuracy_list)
    Bestmodel_avg_test_precision = np.mean(bestmodel_test_precision_list)
    std_auc_test_precision = np.std(bestmodel_test_precision_list)
    Bestmodel_avg_test_recall = np.mean(bestmodel_test_recall_list)
    std_auc_test_recall = np.std(bestmodel_test_recall_list)
    Bestmodel_avg_test_f1 = np.mean(bestmodel_test_f1_list)
    std_auc_test_f1 = np.std(bestmodel_test_f1_list)
    Bestmodel_avg_test_ap = np.mean(bestmodel_test_ap_list)
    std_auc_test_ap = np.std(bestmodel_test_ap_list)

    # Prepare the metrics to be displayed
    bestmodel_final_metrics = {
        "Average Test_auc": (str(Bestmodel_avg_test_auc)[:4]),
        "Average Test_auc_stc": (str(std_auc_test_auc)[:6]),
        "Average Test_accuracy": (str(Bestmodel_avg_test_accuracy)[:4]),
        "Average Test_accuracy_stc": (str(std_auc_test_accuracy)[:6]),
        "Average Test_precision": (str(Bestmodel_avg_test_precision)[:4]),
        "Average Test_precision_stc": (str(std_auc_test_precision)[:6]),
        "Average Test_recall": (str(Bestmodel_avg_test_recall)[:4]),
        "Average Test_recall_stc": (str(std_auc_test_recall)[:6]),
        "Average Test_f1": (str(Bestmodel_avg_test_f1)[:4]),
        "Average Test_f1_stc": (str(std_auc_test_f1)[:6]),
        "Average Test_ap": (str(Bestmodel_avg_test_ap)[:4]),
        "Average Test_ap_stc": (str(std_auc_test_ap)[:6])
    }

    # Display the best metrics in a table 
    for key, value in bestmodel_final_metrics.items():
        bestmode_mean_table.append([key, value])
    print("\nBest Metrics mean for best models over all outer folds:")
    print(tabulate(bestmode_mean_table, headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    main()
