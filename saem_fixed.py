# First import the SAEM class and the numpy module
import numpy as np
from saem import CSEMData

# Set parameters
Min = 300.
max = 3000.
x = -2000
y = 0
Radius = 10

# Load data
data = CSEMData('Tx1run.npz') 
data.detectLines('x')
print("Original frequencies:", data.f)

# Filter data
data.filter(every=3)
print("After filtering:", data)

# Show field data
data.showField("line", label='4', radius=Radius)
data.radius = Radius

# Set parameters for analysis
F = 280  # Hz
freq_index = np.where(data.f == F)[0][0]
Line = 4

print(f"Selected frequency: {data.f[freq_index]} Hz")

# Show data plots
data.showData(line=Line, amphi=True, cmp=[0,0,1])
data.showData(line=Line, amphi=False, cmp=[0,0,1])
data.showData(nf=freq_index, line=Line, cmp=[0,0,1])

# Set sounding position
data.setPos([523200., 1923100], show=True)

# Create depth vector
data.createDepthVector(rho=30)
print("Depth vector:", data.depth)

# Copy Z data to X and Y components
data.DATA[0] = data.DATA[2].copy()
data.DATA[1] = data.DATA[2].copy()

# Copy Z error to X and Y components
data.ERR[0] = data.ERR[2].copy()
data.ERR[1] = data.ERR[2].copy()

# =============================================================================
# FIX FOR inf/nan ERROR VALUES
# =============================================================================

def fix_data_and_errors(data, min_error=1e-6, max_error=1.0):
    """
    Fix inf/nan values in data and ensure reasonable error bounds
    
    Parameters:
    -----------
    data : CSEMData object
        The CSEM data object to fix
    min_error : float
        Minimum allowed error value
    max_error : float
        Maximum allowed error value (as fraction)
    """
    
    print("=== FIXING DATA AND ERRORS ===")
    
    for cmp in range(3):  # X, Y, Z components
        if data.DATA[cmp] is not None and data.ERR[cmp] is not None:
            
            print(f"\nComponent {cmp} ({'XYZ'[cmp]}):")
            
            # Get data and error arrays
            data_vals = data.DATA[cmp]
            error_vals = data.ERR[cmp]
            
            # Check for problems in data
            data_problems = ~np.isfinite(data_vals)
            if np.any(data_problems):
                print(f"  Found {np.sum(data_problems)} inf/nan values in data")
                # Replace with median of finite values
                finite_data = data_vals[np.isfinite(data_vals)]
                if len(finite_data) > 0:
                    replacement_val = np.median(finite_data)
                    data_vals[data_problems] = replacement_val
                    print(f"  Replaced with median value: {replacement_val}")
                else:
                    data_vals[data_problems] = 1e-12
                    print("  Replaced with default small value")
            
            # Check for problems in errors
            error_problems = ~np.isfinite(error_vals) | (error_vals <= 0)
            if np.any(error_problems):
                print(f"  Found {np.sum(error_problems)} inf/nan/zero values in errors")
            
            # Calculate reasonable error bounds
            data_magnitude = np.abs(data_vals[np.isfinite(data_vals)])
            if len(data_magnitude) > 0:
                # Error should be reasonable fraction of data magnitude
                reasonable_error = np.maximum(
                    data_magnitude * 0.05,  # 5% relative error minimum
                    min_error               # Absolute minimum
                )
                
                # For problematic errors, use reasonable estimates
                error_vals[error_problems] = reasonable_error[error_problems] if len(reasonable_error) == len(error_vals) else np.median(reasonable_error)
            
            # Ensure errors are within reasonable bounds
            error_vals = np.clip(error_vals, min_error, np.abs(data_vals) * max_error + min_error)
            
            # Final check: ensure no inf/nan remain
            final_problems = ~np.isfinite(error_vals) | (error_vals <= 0)
            if np.any(final_problems):
                error_vals[final_problems] = min_error
            
            print(f"  Data range: {np.min(data_vals):.2e} to {np.max(data_vals):.2e}")
            print(f"  Error range: {np.min(error_vals):.2e} to {np.max(error_vals):.2e}")
            print(f"  Relative error: {np.min(error_vals/np.abs(data_vals)):.1%} to {np.max(error_vals/np.abs(data_vals)):.1%}")
            
            # Update the data object
            data.DATA[cmp] = data_vals
            data.ERR[cmp] = error_vals

def debug_inversion_data(data, cmp=[0, 0, 1]):
    """
    Debug the data that will be used for inversion
    """
    print("\n=== DEBUGGING INVERSION DATA ===")
    
    # Check which components will be used
    active_components = []
    for i, use_cmp in enumerate(cmp):
        if use_cmp and data.DATA[i] is not None:
            active_components.append(i)
    
    print(f"Active components: {['XYZ'[i] for i in active_components]}")
    
    for cmp_idx in active_components:
        data_vals = data.DATA[cmp_idx].flatten()
        error_vals = data.ERR[cmp_idx].flatten()
        
        print(f"\nComponent {cmp_idx} ({'XYZ'[cmp_idx]}):")
        print(f"  Data points: {len(data_vals)}")
        print(f"  Finite data: {np.sum(np.isfinite(data_vals))}")
        print(f"  Finite errors: {np.sum(np.isfinite(error_vals))}")
        print(f"  Positive errors: {np.sum(error_vals > 0)}")
        print(f"  Data range: {np.min(data_vals):.2e} to {np.max(data_vals):.2e}")
        print(f"  Error range: {np.min(error_vals):.2e} to {np.max(error_vals):.2e}")
        
        # Check for potential problems
        if np.any(~np.isfinite(data_vals)):
            print(f"  WARNING: {np.sum(~np.isfinite(data_vals))} non-finite data values!")
        if np.any(~np.isfinite(error_vals)):
            print(f"  WARNING: {np.sum(~np.isfinite(error_vals))} non-finite error values!")
        if np.any(error_vals <= 0):
            print(f"  WARNING: {np.sum(error_vals <= 0)} zero/negative error values!")

# Apply the fix
fix_data_and_errors(data)

# Debug the data before inversion
debug_inversion_data(data, cmp=[0, 0, 1])

# =============================================================================
# ROBUST INVERSION WITH ERROR HANDLING
# =============================================================================

def robust_invert_sounding(data, **kwargs):
    """
    Perform robust inversion with multiple fallback strategies
    """
    
    # Default parameters
    params = {
        'lam': 10,
        'absError': 0.001,
        'relError': 0.03,
        'cmp': [0, 0, 1],
        'verbose': True,
        'maxIter': 20
    }
    params.update(kwargs)
    
    print(f"\n=== STARTING ROBUST INVERSION ===")
    print(f"Parameters: {params}")
    
    # Strategy 1: Try original parameters
    try:
        print("\nStrategy 1: Original parameters...")
        data.invertSounding(**params)
        print("SUCCESS: Inversion completed with original parameters!")
        return True
        
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Increase absolute error floor
    try:
        print("\nStrategy 2: Higher absolute error floor...")
        params_mod = params.copy()
        params_mod['absError'] = 0.01  # 10x higher
        data.invertSounding(**params_mod)
        print("SUCCESS: Inversion completed with higher error floor!")
        return True
        
    except Exception as e:
        print(f"Strategy 2 failed: {e}")
    
    # Strategy 3: Increase relative error
    try:
        print("\nStrategy 3: Higher relative error...")
        params_mod = params.copy()
        params_mod['relError'] = 0.10  # 10% instead of 3%
        params_mod['absError'] = 0.01
        data.invertSounding(**params_mod)
        print("SUCCESS: Inversion completed with higher relative error!")
        return True
        
    except Exception as e:
        print(f"Strategy 3 failed: {e}")
    
    # Strategy 4: More regularization
    try:
        print("\nStrategy 4: Higher regularization...")
        params_mod = params.copy()
        params_mod['lam'] = 100  # Much higher regularization
        params_mod['relError'] = 0.10
        params_mod['absError'] = 0.01
        data.invertSounding(**params_mod)
        print("SUCCESS: Inversion completed with higher regularization!")
        return True
        
    except Exception as e:
        print(f"Strategy 4 failed: {e}")
    
    # Strategy 5: Only use Z component
    try:
        print("\nStrategy 5: Z-component only...")
        params_mod = params.copy()
        params_mod['cmp'] = [0, 0, 1]  # Only Z
        params_mod['lam'] = 100
        params_mod['relError'] = 0.15
        params_mod['absError'] = 0.05
        data.invertSounding(**params_mod)
        print("SUCCESS: Inversion completed with Z-component only!")
        return True
        
    except Exception as e:
        print(f"Strategy 5 failed: {e}")
    
    print("\nAll inversion strategies failed!")
    return False

# Run robust inversion
success = robust_invert_sounding(data, lam=10, absError=0.001, relError=0.03, cmp=[0, 0, 1], verbose=True)

if success:
    print("\n=== INVERSION RESULTS ===")
    print(f"Model: {data.model}")
    print(f"Response shape: {data.response1d.shape if hasattr(data, 'response1d') else 'N/A'}")
    
    # Plot results if available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot resistivity model
        if hasattr(data, 'model') and data.model is not None:
            resistivity = np.array(data.model)
            depths = data.depth if hasattr(data, 'depth') else np.arange(len(resistivity))
            
            ax1.step(resistivity, depths, '-o', where='post', linewidth=2)
            ax1.set_xscale('log')
            ax1.invert_yaxis()
            ax1.set_xlabel('Resistivity [Ωm]')
            ax1.set_ylabel('Depth [m]')
            ax1.grid(True)
            ax1.set_title('Inverted Resistivity Model')
        
        # Plot data fit
        if hasattr(data, 'response1d') and data.response1d is not None:
            ax2.plot(data.response1d, 'r-', label='Modeled', linewidth=2)
            # Note: observed data would need to be extracted from the data object
            ax2.set_xlabel('Data Point')
            ax2.set_ylabel('Response')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('Data Fit')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Plotting failed: {e}")

else:
    print("\n=== INVERSION FAILED ===")
    print("All strategies failed. Consider:")
    print("1. Checking data quality and preprocessing")
    print("2. Different frequency selection")
    print("3. Alternative inversion approaches")
    print("4. Manual data cleaning")