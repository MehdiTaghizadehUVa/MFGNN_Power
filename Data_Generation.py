import numpy as np
import pypower.api as pp
import torch
from torch_geometric.data import Data
import pickle
import argparse
import os
import time
import networkx as nx

CASE_NAMES = ["case9","case14", "case30", "case118"]
CASE_WEIGHTS = [0.2, 0.2, 0.3, 0.3]

def remove_random_branch(case):
    num_lines = len(case["branch"])
    if num_lines > 1:
        random_branch_idx = np.random.choice(num_lines)
        case["branch"] = np.delete(case["branch"], random_branch_idx, axis=0)


def check_connectivity(case):
    G = nx.Graph()
    for line in case["branch"]:
        G.add_edge(int(line[0]), int(line[1]))
    return nx.is_connected(G)


def generate_and_save_training_data(num_samples, save_path, seed, case_name, power_flow_type):
    np.random.seed(seed)  # Set the random seed for reproducibility

    def perturb_loads(case):
        num_buses = len(case["bus"])

        # Generate perturbations using a normal distribution
        bus_perturbation_p = np.random.normal(1.0, 0.05, size=num_buses)
        bus_perturbation_q = np.random.normal(1.0, 0.05, size=num_buses)

        # Ensure that there are no negative perturbations
        bus_perturbation_p = np.where(bus_perturbation_p < 0, 0.01, bus_perturbation_p)
        bus_perturbation_q = np.where(bus_perturbation_q < 0, 0.01, bus_perturbation_q)

        # Apply perturbations to active and reactive demands
        active_demands = case["bus"][:, 2]
        case["bus"][:, 2] = active_demands.astype(np.float64) * bus_perturbation_p  # Active demand

        reactive_demands = case["bus"][:, 3]
        case["bus"][:, 3] = reactive_demands.astype(np.float64) * bus_perturbation_q  # Reactive demand

    def run_power_flow(case, power_flow_type):
        ppopt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
        try:
            if power_flow_type == "dc":
                results = pp.rundcpf(case, ppopt)
            elif power_flow_type == "ac":
                results = pp.runpf(case, ppopt)
            else:
                raise ValueError("Invalid power flow type. Please specify 'dc' or 'ac'.")

            return results
        except Exception as e:
            print(f"Error running power flow analysis: {e}")
            return None

    # Set the random seed
    np.random.seed(seed)

    # Generate data for the specified number of samples
    successful_samples = 0  # Counter for successful power flow analysis
    data_list = []
    total_time = 0
    while successful_samples < num_samples:
        print(f"Generating Sample {successful_samples + 1}/{num_samples}")

        # Create a sample power system network
        if case_name == "random":
            case_name_i = np.random.choice(CASE_NAMES, p=CASE_WEIGHTS)
        else:
            case_name_i = case_name
        case = getattr(pp, case_name_i)()  # Fetch the specified case using the case_name input
        baseMVA = case["baseMVA"]
        # Randomly perturb load settings
        perturb_loads(case)
        # remove_random_branch(case)

        # if not check_connectivity(case):
        #     print("Disconnected buses detected. Skipping this case.")
        #     continue

        start_time = time.time()
        # Run power flow analysis
        results = run_power_flow(case, power_flow_type)
        end_time = time.time()

        run_time = end_time - start_time

        if results and (np.sum(np.isnan(results[0]['bus'][:, 7])) == 0 and np.sum(np.isnan(results[0]['bus'][:, 8])) == 0):
            total_time += run_time
            bus_active_output = [0] * len(case["bus"])
            bus_reactive_output = [0] * len(case["bus"])
            bus_initial_voltage = [1] * len(case["bus"])

            gen_ids = results[0]['gen'][:, 0]
            gen_active_output = results[0]['gen'][:, 1] / baseMVA
            gen_reactive_output = results[0]['gen'][:, 2] / baseMVA
            gen_voltage_setpoint = results[0]['gen'][:, 5]

            # Convert gen_ids to integers
            gen_ids = gen_ids.astype(int)
            for i, index in enumerate(gen_ids):
                idx = np.where(case["bus"][:, 0] == index)[0][0]
                bus_active_output[idx] = gen_active_output[i]
                bus_reactive_output[idx] = gen_reactive_output[i]
                bus_initial_voltage[idx] = gen_voltage_setpoint[i]

            # Extract voltage magnitudes and active power flows as NumPy arrays
            bus_types = results[0]['bus'][:, 1]
            bus_voltage_magnitudes = results[0]['bus'][:, 7]
            bus_voltage_angles = results[0]['bus'][:, 8] * np.pi / 180
            bus_susceptances = results[0]['bus'][:, 5] / baseMVA
            bus_conductances = results[0]['bus'][:, 4] / baseMVA
            bus_active_demands = results[0]["bus"][:, 2] / baseMVA
            bus_reactive_demands = results[0]["bus"][:, 3] / baseMVA
            bus_active_outputs = bus_active_output
            bus_reactive_outputs = bus_reactive_output
            bus_initial_voltages = bus_initial_voltage

            # Save node IDs and edge start/end nodes
            edge_start_nodes = results[0]['branch'][:, 0]
            edge_end_nodes = results[0]['branch'][:, 1]
            edge_resistance = results[0]['branch'][:, 2]
            edge_reactance = results[0]['branch'][:, 3]
            edge_susceptance = results[0]['branch'][:, 4]
            edge_transformation_ratio = results[0]['branch'][:, 8]
            edge_shift_angle = results[0]['branch'][:, 9] * np.pi / 180

            bus_types = np.array(bus_types)
            bus_active_demands = np.array(bus_active_demands)
            bus_reactive_demands = np.array(bus_reactive_demands)
            bus_active_outputs = np.array(bus_active_outputs)
            bus_reactive_outputs = np.array(bus_reactive_outputs)
            bus_susceptances = np.array(bus_susceptances)
            bus_conductances = np.array(bus_conductances)
            bus_initial_voltages = np.array(bus_initial_voltages)
            bus_voltage_magnitudes = np.array(bus_voltage_magnitudes)
            bus_voltage_angles = np.array(bus_voltage_angles)
            edge_resistance = np.array(edge_resistance)
            edge_reactance = np.array(edge_reactance)
            edge_susceptance = np.array(edge_susceptance)
            edge_transformation_ratio = np.array(edge_transformation_ratio)
            edge_shift_angle = np.array(edge_shift_angle)
            edge_start_nodes = np.array(edge_start_nodes) - 1
            edge_end_nodes = np.array(edge_end_nodes) - 1

            edge_index = np.concatenate((edge_start_nodes.reshape(1, -1), edge_end_nodes.reshape(1, -1)),
                                        axis=0)
            node_features = np.concatenate(
                    (
                        bus_active_demands.reshape(-1, 1),
                        bus_reactive_demands.reshape(-1, 1),
                        bus_active_outputs.reshape(-1, 1),
                        bus_reactive_outputs.reshape(-1, 1),
                        bus_initial_voltages.reshape(-1, 1),
                    ),
                    axis=1,
                )
            edge_features = np.concatenate((edge_resistance.reshape(-1, 1), edge_reactance.reshape(-1, 1), edge_susceptance.reshape(-1, 1), edge_transformation_ratio.reshape(-1, 1), edge_shift_angle.reshape(-1, 1)), axis=1)
            node_targets = np.concatenate((bus_voltage_magnitudes.reshape(-1, 1), bus_voltage_angles.reshape(-1, 1), bus_reactive_outputs.reshape(-1, 1)), axis=1)
            additional_features =np.concatenate((bus_reactive_outputs.reshape(-1, 1), bus_conductances.reshape(-1, 1), bus_susceptances.reshape(-1, 1), bus_types.reshape(-1, 1)), axis=1)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.int64)
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_targets, dtype=torch.float)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            x_add = torch.tensor(additional_features, dtype=torch.float)
            # Create a Data object
            data = Data(x=x, y=y, edge_index=edge_index_tensor, edge_attr=edge_attr, x_add=x_add)
            # data = torch.cat((x, y), dim=-1)
            # Append the Data object to the list
            data_list.append(data)
            successful_samples += 1  # Increment successful samples counter

    # Create a folder based on the case name
    folder = f"data/{case_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = f"{case_name}_{save_path}_{power_flow_type}.pkl"
    folder_save_path = os.path.join(folder, file_name)

    with open(folder_save_path, "wb") as file:
        # Dump the data_list into the file
        pickle.dump(data_list, file)

    print(f"Successfully generated and saved {num_samples} training samples to {file_name}.\n")
    print(f"Average Run Time: {total_time/num_samples}")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Generate and save training data for power system analysis.")

    # Add arguments to the parser
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of power system networks to generate and save.")
    parser.add_argument("--save_path", type=str, default='train_data',
                        help="File path to save the generated data.")
    parser.add_argument("--seed", type=int, default=321, help="Random seed for reproducibility.")
    parser.add_argument("--case_name", type=str, default="case2383wp", help="Name of the power system case to use.")
    parser.add_argument("--power_flow_type", type=str, default="ac", choices=["ac", "dc"],
                        help="Type of power flow analysis. Choose 'dc' for AC power flow or 'dc' for DC power flow.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    generate_and_save_training_data(args.num_samples, args.save_path, args.seed, args.case_name, args.power_flow_type)
