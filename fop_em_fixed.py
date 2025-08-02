# file: fop_em_fixed.py

import numpy as np
import pygimli as pg
import empymod
import matplotlib.pyplot as plt

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

    # Fz1 *= mu0 * 1e9
    return np.hstack((np.real(Fz1), np.imag(Fz1)))

class fopEmpyMod(pg.Modelling):
    def __init__(self, depths):
        super().__init__()
        self.depths = np.abs(depths)
        
        # Create proper 1D mesh for the number of layers
        n_layers = len(self.depths) 
        self.mesh = pg.meshtools.createMesh1D(n_layers)
        self.setMesh(self.mesh)

    def response(self, model):
        # Convert conductivity [S/m] to resistivity [Ohm·m]
        resistivity = 1.0 / np.array(model)
        return fwd_empymod(resistivity, self.depths)

    def createStartModel(self, data):
        # Return conductivity values (not resistivity)
        n_layers = len(self.depths)
        return pg.Vector(n_layers, 0.1)  # Start with 0.1 S/m = 10 Ohm·m

if __name__ == '__main__':
    # Load synthetic data from file
    try:
        data_npz = np.load("mutual_impedance_data (1).npz")
        U = data_npz["Fz1"]
        freqs = data_npz["frequencies"]
        
        data = np.hstack((U.real, U.imag)).astype(np.float64)
        print('Imported data shape:', data.shape)
        print('Imported data:', data)
    except FileNotFoundError:
        print("Data file not found. Creating synthetic data for testing...")
        # Create synthetic data for testing
        true_conductivities = [3.15, 1.0, 5.0, 0.1]  # S/m
        true_depths = [0, 1, 5]
        data = fwd_empymod(1.0/np.array(true_conductivities[1:]), true_depths)
        print('Created synthetic data:', data)

    # Add noise
    np.random.seed(999)
    rel_err = 0.05
    noise_floor = 1e-3
    abs_err = np.abs(data) * rel_err + noise_floor
    d_obs = data + np.random.randn(len(data)) * abs_err
    print('Data + noise shape:', d_obs.shape)
    print('Data + noise:', d_obs)

    # 1. Define model structure - these should be layer boundaries
    depths = [1, 5]  # Layer boundaries (excludes surface at 0)
    
    # 2. Setup forward operator
    fop = fopEmpyMod(depths)
    
    # 3. Create PyGimli DataContainer properly
    data_container = pg.DataContainer()
    
    # Set the number of data points
    n_data = len(d_obs)
    data_container.resize(n_data)
    
    # Add data to container
    for i in range(n_data):
        data_container.set(i, pg.Index(i))  # Set indices
    
    # Set the data values and errors
    data_container.set("rhoa", d_obs)  # Data values
    data_container.set("err", abs_err)  # Error values
    
    # 4. Setup inversion
    inv = pg.Inversion(verbose=True)
    inv.setForwardOperator(fop)
    
    # 5. Start model
    start_model = fop.createStartModel(d_obs)
    print("Start model shape:", len(start_model))
    print("Start model (conductivity):", start_model)
    
    # Test forward response
    response0 = np.array(fop.response(start_model))
    print("Forward response shape:", response0.shape)
    print("Data shape:", d_obs.shape)
    
    chi2_0 = np.mean((response0 - d_obs)**2 / abs_err**2)
    print("Start model χ² (manual):", chi2_0)

    # 6. Set inversion parameters
    inv.setData(data_container)
    inv.setRelativeError(rel_err)
    inv.setAbsoluteError(noise_floor)
    
    # Set regularization
    inv.setLambda(10)  # Start with higher regularization
    inv.setMaxIter(20)
    
    # 7. Run inversion
    try:
        conductivity_model = inv.run(start_model)
        print("Inversion completed successfully!")
        print("Inverted conductivity model:", conductivity_model)
        
        resistivity_model = 1.0 / np.maximum(np.array(conductivity_model), 1e-6)
        print("Inverted resistivity model:", resistivity_model)
        
        # 8. Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Resistivity model
        depth_plot = [0] + depths + [depths[-1] * 2]  # Include surface and extend bottom
        res_plot = np.concatenate(([resistivity_model[0]], resistivity_model, [resistivity_model[-1]]))
        
        ax1.step(res_plot, depth_plot, '-o', where='post', label='Inverted')
        
        # Add true model if available
        if 'true_conductivities' in locals():
            true_res = 1.0 / np.array(true_conductivities)
            true_depth_plot = [0, 1, 5, 10]
            ax1.step(true_res, true_depth_plot, '--x', where='post', label='True Model')
        
        ax1.set_xscale('log')
        ax1.invert_yaxis()
        ax1.set_xlabel("Resistivity [Ωm]")
        ax1.set_ylabel("Depth [m]")
        ax1.grid(True)
        ax1.legend()
        ax1.set_title("1D Resistivity Inversion")
        
        # Plot 2: Data fit
        final_response = np.array(fop.response(conductivity_model))
        ax2.plot(d_obs, 'bo', label='Observed', markersize=4)
        ax2.plot(final_response, 'r-', label='Modeled', linewidth=2)
        ax2.set_xlabel("Data point")
        ax2.set_ylabel("Response")
        ax2.legend()
        ax2.grid(True)
        ax2.set_title("Data Fit")
        
        plt.tight_layout()
        plt.show()
        
        # Print final chi-squared
        final_chi2 = np.mean((final_response - d_obs)**2 / abs_err**2)
        print(f"Final χ²: {final_chi2:.3f}")
        
    except Exception as e:
        print(f"Inversion failed with error: {e}")
        print("Checking dimensions and setup...")
        print(f"Mesh cells: {fop.mesh.cellCount()}")
        print(f"Start model length: {len(start_model)}")
        print(f"Data length: {len(d_obs)}")
        print(f"Forward response length: {len(response0)}")