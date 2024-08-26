import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import TAGConv, GraphConv, GATConv, EdgeConv, SAGEConv, SGConv, APPNP, ChebConv, AGNNConv, GCNConv, GINConv
from models import LFGNN
from torch.optim.lr_scheduler import StepLR
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import logging
import os
import time  # Add the time module for measuring execution time
import random


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set a random seed value
seed = 321
set_random_seeds(seed)


# Define the Low Fidelity GNN model
class LFGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LFGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class HFGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HFGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# Define the High Fidelity GNN model
class MFGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MFGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, additional_x):
        x = torch.cat([x, additional_x], dim=1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x + additional_x


def load_data(file_path, device):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return [d.to(device) for d in data]


def train_low_fidelity_model(model, optimizer, loss_function, train_data, val_data, num_epochs, batch_size, device, save_path):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of each epoch
        total_loss = 0.0
        num_samples = 0

        model.train()
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions = model(x, edge_index)

            loss = loss_function(predictions[:, 0], batch.y[:, 1].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

        avg_loss = total_loss / num_samples
        train_losses.append(avg_loss)

        validation_loss, validation_time = evaluate_low_fidelity(model, val_data, batch_size, device)
        val_losses.append(validation_loss)

        end_time = time.time()  # Record the end time of each epoch
        training_time = (end_time - start_time) / num_samples  # Calculate the time taken for this epoch
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4e}, Validation Loss: {validation_loss:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses, val_losses


def evaluate_low_fidelity(model, data, batch_size, device):
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions = model(x, edge_index)

            loss = mse_loss(predictions[:, 0], batch.y[:, 1].to(device))

            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

    end_time = time.time()
    epoch_time = end_time - start_time

    return total_loss / num_samples, epoch_time / num_samples


def train_high_fidelity_model(model, optimizer, loss_function, train_data, val_data, num_epochs, batch_size, device, save_path):
    train_losses_v = []
    train_losses_d = []
    train_losses = []
    val_losses_v = []
    val_losses_d = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of each epoch
        total_loss_v = 0.0
        total_loss_d = 0.0
        total_loss = 0.0
        num_samples = 0

        model.train()
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions = model(x, edge_index)
            loss_v = loss_function(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = loss_function(predictions[:, 1], batch.y[:, 1].to(device))
            loss = loss_v + loss_d
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

        avg_loss_v = total_loss_v / num_samples
        avg_loss_d = total_loss_d / num_samples
        avg_loss = total_loss / num_samples
        train_losses_v.append(avg_loss_v)
        train_losses_d.append(avg_loss_d)
        train_losses.append(avg_loss)

        validation_loss_v, validation_loss_d, validation_loss, validation_time = evaluate_high_fidelity(model, val_data, batch_size, device)
        val_losses_v.append(validation_loss_v)
        val_losses_d.append(validation_loss_d)
        val_losses.append(validation_loss)

        end_time = time.time()  # Record the end time of each epoch
        training_time = (end_time - start_time) / num_samples  # Calculate the time taken for this epoch
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Validation MSE Loss: {validation_loss:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses_v, val_losses_v, train_losses_d, val_losses_d, train_losses, val_losses


def evaluate_high_fidelity(model, data, batch_size, device):
    model.eval()
    mse_loss_f = nn.MSELoss()
    total_loss_v = 0.0
    total_loss_d = 0.0
    total_loss = 0.0
    num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            predictions = model(x, edge_index)

            loss_v = mse_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = mse_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            loss = loss_v + loss_d

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

    end_time = time.time()
    epoch_time = end_time - start_time

    return total_loss_v / num_samples, total_loss_d / num_samples, total_loss / num_samples, epoch_time / num_samples


def train_multi_fidelity_model(model, optimizer, loss_function, train_data, val_data, num_epochs, batch_size, low_fidelity_model, device, save_path):
    train_losses_v = []
    train_losses_d = []
    train_losses = []
    val_losses_v = []
    val_losses_d = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of each epoch
        total_loss_v = 0.0
        total_loss_d = 0.0
        total_loss = 0.0
        num_samples = 0

        model.train()
        for batch in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)

            with torch.no_grad():
                low_fidelity_predictions = low_fidelity_model(x, edge_index)

            low_fidelity_phase = low_fidelity_predictions[:, 1].view(-1, 1)
            low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

            additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)

            predictions = model(x, edge_index, additional_x)

            loss_v = loss_function(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = loss_function(predictions[:, 1], batch.y[:, 1].to(device))
            loss = loss_v + loss_d
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

        avg_loss_v = total_loss_v / num_samples
        avg_loss_d = total_loss_d / num_samples
        avg_loss = total_loss / num_samples
        train_losses_v.append(avg_loss_v)
        train_losses_d.append(avg_loss_d)
        train_losses.append(avg_loss)


        validation_loss_v, validation_loss_d, validation_loss, validation_time = evaluate_multi_fidelity(model, val_data, batch_size,low_fidelity_model,
                                                                                      device)
        val_losses_v.append(validation_loss_v)
        val_losses_d.append(validation_loss_d)
        val_losses.append(validation_loss)


        end_time = time.time()  # Record the end time of each epoch
        training_time = (end_time - start_time) / num_samples  # Calculate the time taken for this epoch
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train MSE Loss: {avg_loss:.4e}, Validation MSE Loss: {validation_loss:.4e}, Training Time: {training_time:.5e} seconds, Inference Time: {validation_time:.5e} seconds")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    return train_losses_v, val_losses_v, train_losses_d, val_losses_d, train_losses, val_losses


def evaluate_multi_fidelity(model, data, batch_size, low_fidelity_model, device):
    model.eval()
    mse_loss_f = nn.MSELoss()
    total_loss_v = 0.0
    total_loss_d = 0.0
    total_loss = 0.0
    num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)
            low_fidelity_predictions = low_fidelity_model(x, edge_index)

            low_fidelity_phase = low_fidelity_predictions[:, 1].view(-1, 1)
            low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

            additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)

            predictions = model(x, edge_index, additional_x)

            loss_v = mse_loss_f(predictions[:, 0], batch.y[:, 0].to(device))
            loss_d = mse_loss_f(predictions[:, 1], batch.y[:, 1].to(device))
            loss = loss_v + loss_d

            total_loss_v += loss_v.item() * batch.num_graphs
            total_loss_d += loss_d.item() * batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

    end_time = time.time()
    epoch_time = end_time - start_time

    return total_loss_v / num_samples, total_loss_d / num_samples, total_loss / num_samples, epoch_time / num_samples

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Fidelity GNN for Power System Analysis')
    parser.add_argument('--case', type=str, default='case2383wp', help='Name of the power system case (e.g., case118, case14, etc.)')
    parser.add_argument('--input-dir', type=str, default='./data/case2383wp', help='Input directory where data files are located')
    parser.add_argument('--output-dir', type=str, default= 'results', help='Parent directory to save the case folder')
    parser.add_argument('--input-dim', type=int, default=5, help='Input dimension of the data')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for GNN layers')
    parser.add_argument('--output-dim', type=int, default=2, help='Output dimension (e.g., 1 for regression or number of classes for classification)')
    parser.add_argument('--learning-rate-low-fidelity', type=float, default=0.001, help='Learning rate for the Low-Fidelity GNN model')
    parser.add_argument('--learning-rate-high-fidelity', type=float, default=0.001, help='Learning rate for the High-Fidelity GNN model')
    parser.add_argument('--num-epochs-low-fidelity', type=int, default=400, help='Number of epochs for training Low-Fidelity GNN')
    parser.add_argument('--num-epochs-high-fidelity', type=int, default=400, help='Number of epochs for training High-Fidelity GNN')
    parser.add_argument('--batch-size-low-fidelity', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batch-size-high-fidelity', type=int, default=64, help='Batch size for training')
    parser.add_argument('--log-file', type=str, default='training_log.txt', help='File to store the training log')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    return args


def configure_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clear_log_file(log_file):
    with open(log_file, 'w'):
        pass


def main(args, N_MF_TRAIN):
    N_HF_TRAIN = int(1.2 * N_MF_TRAIN)
    N_LF_TRAIN = int(2 * N_MF_TRAIN)

    N_MF_TEST = int(0.5 * N_MF_TRAIN)
    N_HF_TEST = int(0.5 * N_MF_TRAIN)
    N_LF_TEST = int(0.5 * N_LF_TRAIN)

    N_MF_VAL = int(0.25 * N_MF_TRAIN)
    N_HF_VAL = int(0.25 * N_MF_TRAIN)
    N_LF_VAL = int(0.25 * N_LF_TRAIN)

    # args.output_dir = f'./Sensitivity/results_{N_MF_TRAIN}'
    # Create the case folder with the specified folder name
    case_folder = os.path.join(args.output_dir, args.case)
    os.makedirs(case_folder, exist_ok=True)
    # Clear existing log file or create a new one
    log_file_path = os.path.join(case_folder, f'training_log.txt')
    clear_log_file(log_file_path)

    # Configure logging to write to the log file
    configure_logging(log_file_path)

    logging.info(f"***********************RESULTS FOR NUMBER OF HIGH-FIDELITY DATA = {N_MF_TRAIN}***********************")
    # Load the data
    train_data_DC = load_data(os.path.join(args.input_dir, f'{args.case}_train_data_dc.pkl'), args.device)
    validation_data_DC = load_data(os.path.join(args.input_dir, f'{args.case}_val_data_dc.pkl'), args.device)
    # test_data_dc = load_data(os.path.join(args.input_dir, f'{args.case}_test_data_dc.pkl'), args.device)
    train_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_train_data_ac.pkl'), args.device)
    validation_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_val_data_ac.pkl'), args.device)
    test_data_AC = load_data(os.path.join(args.input_dir, f'{args.case}_test_data_ac.pkl'), args.device)

    train_data_ac_mf = random.sample(train_data_AC, N_MF_TRAIN)
    train_data_ac = random.sample(train_data_AC, N_HF_TRAIN)
    train_data_dc = random.sample(train_data_DC, N_LF_TRAIN)

    validation_data_ac = random.sample(validation_data_AC, N_HF_VAL)
    validation_data_dc = random.sample(validation_data_DC, N_LF_VAL)
    test_data_ac = random.sample(test_data_AC, N_HF_TEST)



    # Create Low-Fidelity and High-Fidelity models
    low_fidelity_model = LFGNN(args.input_dim, args.hidden_dim, args.output_dim).to(args.device)
    high_fidelity_model = MFGNN(args.input_dim + args.output_dim, args.hidden_dim, args.output_dim).to(args.device)

    # Define optimizers and loss function
    low_fidelity_optimizer = optim.Adam(low_fidelity_model.parameters(), lr=args.learning_rate_low_fidelity)
    high_fidelity_optimizer = optim.Adam(high_fidelity_model.parameters(), lr=args.learning_rate_high_fidelity)
    loss_function = nn.MSELoss()

    # Training the Low-Fidelity GNN model
    logging.info("Training Low-Fidelity GNN...")
    low_fidelity_model_save_path = os.path.join(case_folder, 'LF_best_model.pt')
    train_losses_low_fidelity, val_losses_low_fidelity = train_low_fidelity_model(low_fidelity_model, low_fidelity_optimizer, loss_function, train_data_dc, validation_data_dc, args.num_epochs_low_fidelity, args.batch_size_low_fidelity, args.device, low_fidelity_model_save_path)

    # Load the best Low-Fidelity model based on validation losses
    low_fidelity_model.load_state_dict(torch.load(low_fidelity_model_save_path))


    # Training the High-Fidelity GNN model using Low-Fidelity predictions
    logging.info("\nTraining Multi-Fidelity GNN...")
    multi_fidelity_model_save_path = os.path.join(case_folder, 'MF_best_model.pt')
    train_losses_v_high_fidelity, val_losses_v_high_fidelity, train_losses_d_high_fidelity, val_losses_d_high_fidelity, train_losses_high_fidelity, val_losses_high_fidelity, = train_multi_fidelity_model(high_fidelity_model,
                                                                                     high_fidelity_optimizer,
                                                                                     loss_function, train_data_ac_mf,
                                                                                     validation_data_ac,
                                                                                     args.num_epochs_high_fidelity,
                                                                                     args.batch_size_high_fidelity,
                                                                                     low_fidelity_model, args.device,
                                                                                     multi_fidelity_model_save_path)

    # Load the best High-Fidelity model based on validation losses
    high_fidelity_model.load_state_dict(torch.load(multi_fidelity_model_save_path))

    # Evaluation of the High-Fidelity model on the test set
    logging.info("\nEvaluating Multi-Fidelity GNN on Test Set...")
    test_loss_v_multi_fidelity, test_loss_d_multi_fidelity, test_loss_multi_fidelity, test_time_multi_fidelity = evaluate_multi_fidelity(high_fidelity_model, test_data_ac, args.batch_size_high_fidelity, low_fidelity_model, args.device)
    logging.info(f"Test MSE Loss - Magnitude: {test_loss_v_multi_fidelity:.5e}")
    logging.info(f"Test MSE Loss - Phase: {test_loss_d_multi_fidelity:.5e}")
    logging.info(f"Test MSE Loss: {test_loss_multi_fidelity:.5e}")
    logging.info(f"Test Inference Time: {test_time_multi_fidelity:.5e}")

    # Train the High-Fidelity GNN model using HF data only
    high_fidelity_model_only_hf = HFGNN(args.input_dim, args.hidden_dim, args.output_dim).to(args.device)
    high_fidelity_optimizer_only_hf = optim.Adam(high_fidelity_model_only_hf.parameters(), lr=args.learning_rate_high_fidelity)

    # Training loop for High-Fidelity GNN using HF data only
    logging.info("\nTraining High-Fidelity GNN (HF data only)...")
    high_fidelity_model_only_hf_save_path = os.path.join(case_folder, 'HF_best_model.pt')
    train_losses_v_high_fidelity_only_hf, val_losses_v_high_fidelity_only_hf, train_losses_d_high_fidelity_only_hf, val_losses_d_high_fidelity_only_hf, train_losses_high_fidelity_only_hf, val_losses_high_fidelity_only_hf= train_high_fidelity_model(high_fidelity_model_only_hf, high_fidelity_optimizer_only_hf, loss_function, train_data_ac, validation_data_ac, args.num_epochs_high_fidelity, args.batch_size_high_fidelity, args.device, high_fidelity_model_only_hf_save_path)

    # Load the best High-Fidelity model based on validation losses
    high_fidelity_model_only_hf.load_state_dict(torch.load(high_fidelity_model_only_hf_save_path))

    # Evaluation of High-Fidelity GNN model using HF data only on the test set
    logging.info("\nEvaluating High-Fidelity GNN (HF data only) on Test Set...")
    test_loss_v_hf_only, test_loss_d_hf_only, test_loss_hf_only, test_time_hf_only = evaluate_high_fidelity(high_fidelity_model_only_hf, test_data_ac, args.batch_size_high_fidelity, args.device)
    logging.info(f"Test MSE Loss - Magnitude (HF data only): {test_loss_v_hf_only:.5e}")
    logging.info(f"Test MSE Loss- Phase (HF data only): {test_loss_d_hf_only:.5e}")
    logging.info(f"Test MSE Loss (HF data only): {test_loss_hf_only:.5e}")
    logging.info(f"Test Inference Time: {test_time_hf_only:.5e}")

    # Compare the losses
    logging.info("\nComparison of Test MSE Loss - Magnitude:")
    logging.info(f"Multi-Fidelity Approach: {test_loss_v_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_loss_v_hf_only:.5e}")

    logging.info("\nComparison of Test MSE Loss - Phase:")
    logging.info(f"Multi-Fidelity Approach: {test_loss_d_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_loss_d_hf_only:.5e}")

    logging.info("\nComparison of Test MSE Loss:")
    logging.info(f"Multi-Fidelity Approach: {test_loss_multi_fidelity:.5e}")
    logging.info(f"HF Data Only Approach:  {test_loss_hf_only:.5e}")


    # Plotting for HF Data Only Approach
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24}
    plt.rc('font', **font)
    plt.figure(figsize=(12, 7.4))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_v_high_fidelity_only_hf, label='High-Fidelity Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_v_high_fidelity_only_hf, label='High-Fidelity Validation Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_v_high_fidelity, label='Multi-Fidelity Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_v_high_fidelity, label='Multi-Fidelity Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_losses_magnitude.png'), dpi=700)
    # Close the plot to release resources
    plt.close()

    plt.figure(figsize=(12, 7.4))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_d_high_fidelity_only_hf, label='High-Fidelity Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_d_high_fidelity_only_hf, label='High-Fidelity Validation Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_d_high_fidelity, label='Multi-Fidelity Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_d_high_fidelity, label='Multi-Fidelity Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_losses_phase.png'), dpi=700)
    # Close the plot to release resources
    plt.close()

    plt.figure(figsize=(12, 7.4))
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_high_fidelity_only_hf, label='High-Fidelity Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_high_fidelity_only_hf, label='High-Fidelity Validation Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), train_losses_high_fidelity, label='Multi-Fidelity Train Loss')
    plt.plot(range(1, args.num_epochs_high_fidelity + 1), val_losses_high_fidelity, label='Multi-Fidelity Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(case_folder, 'multi_fidelity_mse_losses.png'), dpi=700)
    # Close the plot to release resources
    plt.close()


    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_v_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_v_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_d_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_d_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_train_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_high_fidelity, f)

    output_data_path = os.path.join(case_folder, 'mf_eval_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_high_fidelity, f)


    output_data_path = os.path.join(case_folder, 'hf_train_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_v_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_eval_mse_losses_magnitude.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_v_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_train_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_d_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_eval_mse_losses_phase.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_d_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_train_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(train_losses_high_fidelity_only_hf, f)

    output_data_path = os.path.join(case_folder, 'hf_eval_mse_losses.pkl')
    with open(output_data_path, 'wb') as f:
        pickle.dump(val_losses_high_fidelity_only_hf, f)


    # Visualization of the power system graph with node colors representing the prediction errors
    selected_sample_index = np.random.randint(len(test_data_ac))
    selected_sample = test_data_ac[selected_sample_index]

    x, edge_index = selected_sample.x.to(args.device), selected_sample.edge_index.to(args.device)
    with torch.no_grad():
        low_fidelity_predictions = low_fidelity_model(x, edge_index)

    low_fidelity_phase = low_fidelity_predictions[:, 1].view(-1, 1)
    low_fidelity_voltage = torch.ones_like(low_fidelity_phase)

    additional_x = torch.cat([low_fidelity_voltage, low_fidelity_phase], dim=1)
    high_fidelity_model.eval()
    with torch.no_grad():
        predicted_output = high_fidelity_model(x, edge_index, additional_x)

    true_output = selected_sample.y.cpu().numpy()
    errors_v = np.abs(predicted_output[:, 0].cpu().numpy() - true_output[:, 0])
    errors_d = np.abs(predicted_output[:, 1].cpu().numpy() - true_output[:, 1])

    graph = nx.Graph()
    for node_idx in range(selected_sample.num_nodes):
        node_value = selected_sample.x[node_idx].cpu().numpy()
        graph.add_node(node_idx, value=node_value)

    for edge_idx in range(selected_sample.edge_index.shape[1]):
        src, tgt = selected_sample.edge_index[:, edge_idx].cpu().numpy()
        graph.add_edge(src, tgt)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    node_colors = [errors_v[node_idx] for node_idx in graph.nodes()]
    edges = nx.draw_networkx_edges(graph, pos, edge_color='gray')
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap='rainbow', node_size=250, alpha=0.9)
    plt.colorbar(nodes, label='Absolute Error')
    labels = nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')
    plt.axis('off')
    plt.title("Voltage Magnitude")
    plt.savefig(os.path.join(case_folder, f"network_error_magnitude_{selected_sample_index}.png"), dpi=700)
    # Close the plot to release resources
    plt.close()

    plt.figure(figsize=(12, 8))
    node_colors = [errors_d[node_idx] for node_idx in graph.nodes()]
    edges = nx.draw_networkx_edges(graph, pos, edge_color='gray')
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap='rainbow', node_size=250, alpha=0.9)
    plt.colorbar(nodes, label='Absolute Error')
    labels = nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')
    plt.axis('off')
    plt.title("Phase Angle")
    plt.savefig(os.path.join(case_folder, f"network_error_phase_{selected_sample_index}.png"), dpi=700)
    # Close the plot to release resources
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    # for N_MF_TRAIN in range(8000, 20001, 2000):
    main(args, N_MF_TRAIN =14000)
