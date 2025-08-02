# file: fop_em_multi_data.py

import numpy as np
import pygimli as pg
import empymod
import matplotlib.pyplot as plt
import os
import glob

def fwd_empymod(resistivities, depths):
    # sea air layer on top
    resistivities = np.concatenate(([1/3.15], resistivities))
    depths = np.concatenate(([0.0], depths))

    frequencies = np.array([15., 90., 330., 810., 1590., 2850., 5010., 8970.])
    radius_main = 3.34 / 2
    H = 0
    tx_height = H - 0.28
    rec_height = tx_height
    mu0 = 4 * np.pi * 1e-7
    n_turns_tx = 8

    Fz1 = np.zeros(len(frequencies), dtype=complex)

    for i, freq in enumerate(frequencies):
        Fz1[i] = empymod.bipole(
            src=[radius_main, 0, tx_height, 90, 0],
            rec=[0, 0, rec_height, 0, 90],
            depth=depths,
            res=resistivities,
            freqtime=freq,
            strength=n_turns_tx * 2 * np.pi * radius_main,
            mrec=True,
            msrc=False,
            srcpts=1,
            verb=0,
        )

    return np.hstack((np.real(Fz1), np.imag(Fz1)))

class fopEmpyModMulti(pg.Modelling):
    def __init__(self, depths, n_datasets):
        super().__init__()
        self.depths = np.abs(depths)
        self.n_datasets = n_datasets
        
        # Create proper 1D mesh for the number of layers
        n_layers = len(self.depths) + 1  # +1 because depths are boundaries
        self.mesh = pg.meshtools.createMesh1D(n_layers)
        self.setMesh(self.mesh)

    def response(self, model):
        # Convert conductivity [S/m] to resistivity [Ohm·m]
        # Constrain conductivity values to reasonable range
        model_constrained = np.clip(np.array(model), 1e-3, 10.0)  # 1mS/m to 10S/m
        resistivity = 1.0 / model_constrained
        
        # Calculate forward response for single location
        single_response = fwd_empymod(resistivity, self.depths)
        
        # Repeat the response for all datasets (assuming same subsurface structure)
        # In practice, you might want different forward operators for different locations
        full_response = np.tile(single_response, self.n_datasets)
        
        return full_response

    def createStartModel(self, data):
        # Return conductivity values (not resistivity)
        n_layers = len(self.depths) + 1
        return pg.Vector(n_layers, 0.1)  # Start with 0.1 S/m = 10 Ohm·m

def load_multiple_data_files(file_pattern="mutual_impedance_data*.npz"):
    """
    Load multiple data files and combine them
    
    Parameters:
    -----------
    file_pattern : str
        Pattern to match data files (e.g., "data*.npz" or specific filenames)
    
    Returns:
    --------
    combined_data : numpy array
        Combined data from all files
    combined_errors : numpy array
        Combined error estimates
    file_list : list
        List of loaded files
    """
    
    # Find all matching files
    data_files = glob.glob(file_pattern)
    
    if not data_files:
        print(f"No files found matching pattern: {file_pattern}")
        return None, None, []
    
    print(f"Found {len(data_files)} data files: {data_files}")
    
    all_data = []
    all_files = []
    
    for file_path in data_files:
        try:
            print(f"Loading {file_path}...")
            data_npz = np.load(file_path)
            
            # Extract data (adjust keys based on your file structure)
            if "Fz1" in data_npz:
                U = data_npz["Fz1"]
            elif "data" in data_npz:
                U = data_npz["data"]
            else:
                print(f"Warning: No recognized data key in {file_path}")
                continue
                
            # Convert to real/imaginary format
            data = np.hstack((U.real, U.imag)).astype(np.float64)
            all_data.append(data)
            all_files.append(file_path)
            
            print(f"  Loaded data shape: {data.shape}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        return None, None, []
    
    # Combine all data
    combined_data = np.concatenate(all_data)
    
    # Create error estimates for combined data
    rel_err = 0.05
    noise_floor = 1e-3
    combined_errors = np.abs(combined_data) * rel_err + noise_floor
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Total number of datasets: {len(all_data)}")
    
    return combined_data, combined_errors, all_files

if __name__ == '__main__':
    
    # Method 1: Load multiple files automatically
    print("=== LOADING MULTIPLE DATA FILES ===")
    
    # Try different file patterns
    file_patterns = [
        "mutual_impedance_data*.npz",  # Your original pattern
        "*.npz",                       # All .npz files
        "data*.npz"                    # Alternative pattern
    ]
    
    combined_data = None
    combined_errors = None
    file_list = []
    
    for pattern in file_patterns:
        combined_data, combined_errors, file_list = load_multiple_data_files(pattern)
        if combined_data is not None:
            break
    
    # Method 2: Manual file specification (if automatic doesn't work)
    if combined_data is None:
        print("\nAutomatic loading failed. Trying manual file specification...")
        
        # Specify your files manually here
        manual_files = [
            "mutual_impedance_data (1).npz",
            # Add more files here:
            # "mutual_impedance_data (2).npz",
            # "mutual_impedance_data (3).npz",
        ]
        
        all_data = []
        all_files = []
        
        for file_path in manual_files:
            if os.path.exists(file_path):
                try:
                    data_npz = np.load(file_path)
                    U = data_npz["Fz1"]
                    data = np.hstack((U.real, U.imag)).astype(np.float64)
                    all_data.append(data)
                    all_files.append(file_path)
                    print(f"Loaded {file_path}: shape {data.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if all_data:
            combined_data = np.concatenate(all_data)
            rel_err = 0.05
            noise_floor = 1e-3
            combined_errors = np.abs(combined_data) * rel_err + noise_floor
            file_list = all_files
    
    # Method 3: Create synthetic multi-dataset for testing
    if combined_data is None:
        print("\nNo data files found. Creating synthetic multi-dataset for testing...")
        
        # Create multiple synthetic datasets with slight variations
        true_conductivities_list = [
            [3.15, 1.0, 5.0, 0.1],      # Dataset 1
            [3.15, 1.2, 4.5, 0.15],     # Dataset 2 (slightly different)
            [3.15, 0.8, 5.5, 0.08],     # Dataset 3 (slightly different)
        ]
        
        true_depths = [1, 5]
        all_data = []
        
        for i, true_cond in enumerate(true_conductivities_list):
            data = fwd_empymod(1.0/np.array(true_cond[1:]), true_depths)
            all_data.append(data)
            print(f"Created synthetic dataset {i+1}: shape {data.shape}")
        
        combined_data = np.concatenate(all_data)
        rel_err = 0.05
        noise_floor = 1e-3
        combined_errors = np.abs(combined_data) * rel_err + noise_floor
        file_list = [f"synthetic_dataset_{i+1}" for i in range(len(all_data))]
    
    # Add noise to combined data
    np.random.seed(999)
    d_obs = combined_data + np.random.randn(len(combined_data)) * combined_errors
    
    print(f"\n=== FINAL COMBINED DATA ===")
    print(f"Total data points: {len(d_obs)}")
    print(f"Number of datasets: {len(file_list)}")
    print(f"Data shape: {d_obs.shape}")
    
    # 1. Define model structure
    depths = [1, 5]  # Layer boundaries (excludes surface at 0)
    n_datasets = len(file_list)
    
    # 2. Setup forward operator for multiple datasets
    fop = fopEmpyModMulti(depths, n_datasets)
    
    # 3. Start model
    start_model = fop.createStartModel(d_obs)
    print(f"Start model shape: {len(start_model)}")
    print(f"Start model (conductivity): {start_model}")
    
    # Test forward response
    response0 = np.array(fop.response(start_model))
    print(f"Forward response shape: {response0.shape}")
    print(f"Data shape: {d_obs.shape}")
    
    chi2_0 = np.mean((response0 - d_obs)**2 / combined_errors**2)
    print(f"Start model χ² (manual): {chi2_0}")

    # 4. Setup inversion
    inv = pg.Inversion(verbose=True)
    inv.setForwardOperator(fop)
    
    # Set inversion parameters - might need more regularization with more data
    inv.setLambda(1.0)  # Regularization
    inv.setMaxIter(20)
    inv.setDeltaPhiAbortPercent(1.0)
    
    # 5. Run inversion
    try:
        conductivity_model = inv.run(d_obs, combined_errors, startModel=start_model)
        
        print("Multi-dataset inversion completed successfully!")
        print("Inverted conductivity model:", conductivity_model)
        
        # Ensure reasonable bounds for plotting
        conductivity_array = np.array(conductivity_model)
        conductivity_array = np.clip(conductivity_array, 1e-6, 100)
        resistivity_model = 1.0 / conductivity_array
        
        print("Inverted resistivity model:", resistivity_model)
        
        # 6. Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Resistivity model
        if len(depths) == 2:
            depth_boundaries = [0] + depths + [depths[-1] * 2]
            res_for_plot = list(resistivity_model) + [resistivity_model[-1]]
            
            ax1.step(res_for_plot, depth_boundaries, '-o', where='post', 
                    label='Inverted (Multi-data)', linewidth=3, markersize=8)
        
        # Add true model if available
        if 'true_conductivities_list' in locals():
            for i, true_cond in enumerate(true_conductivities_list):
                true_res = 1.0 / np.array(true_cond)
                true_depth_plot = [0, 1, 5, 10]
                ax1.step(true_res, true_depth_plot, '--', alpha=0.7,
                        label=f'True Model {i+1}', linewidth=2)
        
        ax1.set_xscale('log')
        ax1.invert_yaxis()
        ax1.set_xlabel("Resistivity [Ωm]")
        ax1.set_ylabel("Depth [m]")
        ax1.grid(True)
        ax1.legend()
        ax1.set_title("1D Resistivity Inversion (Multi-Dataset)")
        ax1.set_xlim([0.1, 1000])
        
        # Plot 2: Combined data fit
        final_response = np.array(fop.response(conductivity_model))
        
        data_per_set = len(combined_data) // n_datasets
        n_freq = data_per_set // 2
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i in range(n_datasets):
            start_idx = i * data_per_set
            end_idx = (i + 1) * data_per_set
            
            obs_data = d_obs[start_idx:end_idx]
            model_data = final_response[start_idx:end_idx]
            
            freq_indices = np.arange(n_freq) + i * 0.1  # Slight offset for visibility
            color = colors[i % len(colors)]
            
            ax2.plot(freq_indices, obs_data[:n_freq], 'o', color=color, alpha=0.7,
                    label=f'Obs Real {i+1}', markersize=3)
            ax2.plot(freq_indices, model_data[:n_freq], '-', color=color,
                    label=f'Mod Real {i+1}', linewidth=2)
        
        ax2.set_xlabel("Frequency index")
        ax2.set_ylabel("Response (Real)")
        ax2.legend()
        ax2.grid(True)
        ax2.set_title("Data Fit - Real Part")
        
        # Plot 3: Imaginary part
        for i in range(n_datasets):
            start_idx = i * data_per_set
            end_idx = (i + 1) * data_per_set
            
            obs_data = d_obs[start_idx:end_idx]
            model_data = final_response[start_idx:end_idx]
            
            freq_indices = np.arange(n_freq) + i * 0.1
            color = colors[i % len(colors)]
            
            ax3.plot(freq_indices, obs_data[n_freq:], 'o', color=color, alpha=0.7,
                    label=f'Obs Imag {i+1}', markersize=3)
            ax3.plot(freq_indices, model_data[n_freq:], '-', color=color,
                    label=f'Mod Imag {i+1}', linewidth=2)
        
        ax3.set_xlabel("Frequency index")
        ax3.set_ylabel("Response (Imaginary)")
        ax3.legend()
        ax3.grid(True)
        ax3.set_title("Data Fit - Imaginary Part")
        
        # Plot 4: Chi-squared per dataset
        chi2_per_dataset = []
        for i in range(n_datasets):
            start_idx = i * data_per_set
            end_idx = (i + 1) * data_per_set
            
            obs_subset = d_obs[start_idx:end_idx]
            model_subset = final_response[start_idx:end_idx]
            error_subset = combined_errors[start_idx:end_idx]
            
            chi2 = np.mean((model_subset - obs_subset)**2 / error_subset**2)
            chi2_per_dataset.append(chi2)
        
        ax4.bar(range(1, n_datasets+1), chi2_per_dataset, alpha=0.7)
        ax4.set_xlabel("Dataset Number")
        ax4.set_ylabel("Chi-squared")
        ax4.set_title("Chi-squared per Dataset")
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        final_chi2 = np.mean((final_response - d_obs)**2 / combined_errors**2)
        print(f"\n=== INVERSION RESULTS ===")
        print(f"Final χ² (overall): {final_chi2:.3f}")
        print(f"RMS error: {np.sqrt(np.mean((final_response - d_obs)**2)):.6f}")
        print(f"Number of iterations: {inv.iter()}")
        
        print(f"\n=== CHI-SQUARED PER DATASET ===")
        for i, chi2 in enumerate(chi2_per_dataset):
            dataset_name = file_list[i] if i < len(file_list) else f"Dataset {i+1}"
            print(f"{dataset_name}: χ² = {chi2:.3f}")
        
        # Print layer interpretation
        print("\n=== LAYER INTERPRETATION ===")
        for i, (res, cond) in enumerate(zip(resistivity_model, conductivity_model)):
            if i == 0:
                depth_range = f"0 - {depths[0]} m"
            elif i == len(resistivity_model) - 1:
                depth_range = f"{depths[i-1]} m - ∞"
            else:
                depth_range = f"{depths[i-1]} - {depths[i]} m"
            
            print(f"Layer {i+1} ({depth_range}): {res:.2f} Ωm ({cond:.4f} S/m)")
        
    except Exception as e:
        print(f"Multi-dataset inversion failed with error: {e}")
        print("Checking dimensions and setup...")
        print(f"Mesh cells: {fop.mesh.cellCount()}")
        print(f"Start model length: {len(start_model)}")
        print(f"Data length: {len(d_obs)}")
        print(f"Forward response length: {len(response0)}")
        
        import traceback
        traceback.print_exc()