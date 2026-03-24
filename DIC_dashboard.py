import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. PAGE CONFIGURATION & GUI SETUP
# ==========================================
st.set_page_config(page_title="DIC Wave Calibration", layout="wide")
st.title("DIC measurement validation tool")
st.markdown("Generate theoretical DIC displacement fields and validate experimental measurements.")

# --- IN-APP DOCUMENTATION FOR THE DIC EXPERT ---
with st.expander("READ ME: Coordinate System & Sign Conventions (Click to expand)"):
    st.markdown("""
    ### Coordinate System
    * **Origin (0,0,0):** Located exactly at the optical center of the camera line-of-sight, on the **resting water surface**.
    * **Z-Axis:** Points UP. The camera is at $Z = +H_{CAM}$. The tank floor is at $Z = -H_{BOT}$.
    * **Grid (X, Y):** Grid of evaluation points. I need 256x256 displacement vectors defined on the 20x20m area on the tank floor. 
    
    ### Matrix Indexing
    The `.npy` matrix must use a standard Cartesian layout, **not** a top-down image layout. For a 256x256 grid over a 20x20m bounds area:
    * **`matrix[0, 0]`** = Bottom-Left corner `(X = -10m, Y = -10m)`
    * **`matrix[255, 0]`** = Top-Left corner `(X = -10m, Y = +10m)`
    * **`matrix[0, 255]`** = Bottom-Right corner `(X = +10m, Y = -10m)`
    * **`matrix[255, 255]`** = Top-Right corner `(X = +10m, Y = +10m)`
    * The 3rd dimension holds the flow: `matrix[:, :, 0]` is $dX$ and `matrix[:, :, 1]` is $dY$.
    
    ### Sign Conventions for Displacement (dX, dY)
    * $dX$ and $dY$ measure the **apparent optical shift** of the pattern on the tank floor.
    * **Calculation:** $dX = X_{wavy} - X_{flat}$. Deformed image by waves minus reference image with flat water.
    * **Meaning:** A positive $dX$ (+1.5 mm) means that due to the wave, the line-of-sight passing through that specific grid point hits a speckle on the tank floor that is physically located 1.5 mm further in the positive X direction compared to flat water.
    * **DIC Software Alignment:** Ensure your DIC software outputs `Deformed - Reference`, not `Reference - Deformed`.
    
    ### Physical Units
    * All spatial inputs and outputs (X, Y, dX, dY, Elevation) are strictly in **METERS**.
    """)

# ==========================================
# 2. SIDEBAR CONFIGURATION (User Inputs)
# ==========================================
st.sidebar.header("1. Facility Geometry")
# Note: H_BOT is negative because the origin Z=0 is the resting water surface.
H_CAM = st.sidebar.number_input("Camera Height (+Z m)", value=8.0, step=0.5)
H_BOT = st.sidebar.number_input("Tank Bottom Depth (-Z m)", value=-2.0, step=0.5)
N_AIR = st.sidebar.number_input("Refractive Index (Air)", value=1.0003, format="%.4f")
N_WATER = st.sidebar.number_input("Refractive Index (Water)", value=1.3330, format="%.4f")

st.sidebar.header("2. Grid Parameters")
BOUNDS = st.sidebar.number_input("Half-width Bounds (m)", value=10.0, step=1.0)
GRID_SIZE = st.sidebar.number_input("Grid Size (Points)", value=256, step=64)

st.sidebar.header("3. Harmonic Wave")
AMPLITUDE = st.sidebar.number_input("Amplitude (m)", value=0.05, step=0.01)
WAVELENGTH = st.sidebar.number_input("Wavelength (m)", value=3.0, step=0.5)
DIRECTION = st.sidebar.number_input("Direction (degrees)", value=0.0, step=5.0)
PHASE = st.sidebar.number_input("Phase (radians)", value=0.0, step=0.1)

# ==========================================
# 3. PHYSICS SOLVER (Theoretical Ray-Tracing)
# ==========================================

def get_harmonic_wave(x, y, amplitude, wavelength, direction, phase):
    """
    Calculates the true physical elevation and surface slopes of the wave.
    Uses standard linear wave theory kinematics.
    """
    # Wavenumber magnitude
    k = 2.0 * np.pi / wavelength
    dir_rad = np.radians(direction)
    
    # Wavenumber vector components
    kx, ky = k * np.cos(dir_rad), k * np.sin(dir_rad)
    
    # Phase argument: k \cdot x - \omega t + \phi. (Time is frozen, so just spatial + phase)
    arg = kx * x + ky * y + phase
    
    # Surface Elevation (eta)
    eta = amplitude * np.cos(arg)
    
    # Surface Slopes (Partial derivatives of elevation w.r.t X and Y)
    eta_dx = -amplitude * kx * np.sin(arg)
    eta_dy = -amplitude * ky * np.sin(arg)
    
    return eta, eta_dx, eta_dy

@st.cache_data
def generate_theoretical_flow(h_cam, h_bot, bounds, grid_size, n_air, n_wat, amp, wave_l, dir_deg, phase_rad):
    """
    Simulates the DIC process by tracing light rays from the camera, through the 
    wavy water interface, down to the tank floor, and comparing against a flat-water reference.
    """
    # 1. Define the Eulerian coordinate grid at the resting water surface (Z=0)
    x_coords = np.linspace(-bounds, bounds, grid_size)
    y_coords = np.linspace(-bounds, bounds, grid_size)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 2. Camera Incident Rays
    # Calculate the normalized direction vector (v_air) of the light ray from the camera down to the grid.
    d_norm = np.sqrt(X**2 + Y**2 + h_cam**2)
    v_air_x, v_air_y, v_air_z = X / d_norm, Y / d_norm, -h_cam / d_norm
    
    # 3. Ray-Surface Intersection (Iterative)
    # Because the water is wavy, the ray doesn't hit the water exactly at Z=0.
    # We iteratively find the exact 3D coordinate (x_surf, y_surf, z_surf) where the ray hits the wave.
    s = np.ones_like(X) # Initial scale guess
    for _ in range(5):
        eta, _, _ = get_harmonic_wave(s * X, s * Y, amp, wave_l, dir_deg, phase_rad)
        s = 1.0 - eta / h_cam # Adjust scale based on wave height at guess location
        
    x_surf, y_surf, z_surf = s * X, s * Y, h_cam * (1 - s)
    
    # 4. Surface Normal Vector
    # Calculate the 3D normal vector of the water surface at the exact intersection point.
    _, eta_dx, eta_dy = get_harmonic_wave(x_surf, y_surf, amp, wave_l, dir_deg, phase_rad)
    norm_factor = np.sqrt(eta_dx**2 + eta_dy**2 + 1.0)
    # Normal points UP out of the water
    n_x, n_y, n_z = -eta_dx / norm_factor, -eta_dy / norm_factor, 1.0 / norm_factor
    
    # 5. Snell's Law (3D Vector Formulation)
    # Calculates the new direction vector (v_wat) of the ray after it bends into the water.
    r = n_air / n_wat
    c = -(v_air_x * n_x + v_air_y * n_y + v_air_z * n_z) # Cosine of angle of incidence
    factor = r * c - np.sqrt(1.0 - r**2 * (1.0 - c**2))
    
    v_wat_x = r * v_air_x + factor * n_x
    v_wat_y = r * v_air_y + factor * n_y
    v_wat_z = r * v_air_z + factor * n_z
    
    # 6. Wavy Water Bottom Intersection (x_bot)
    # Trace the refracted ray through the water until it hits the tank floor (H_BOT).
    tau = (h_bot - z_surf) / v_wat_z # Distance travel multiplier
    x_bot = x_surf + tau * v_wat_x
    y_bot = y_surf + tau * v_wat_y 
    
    # 7. Flat Water Bottom Intersection (x_ref_bot) -> The "Reference Image"
    # Recalculate where the EXACT same camera ray would land if the water was perfectly flat (Z=0).
    c_flat = -v_air_z 
    factor_flat = r * c_flat - np.sqrt(1.0 - r**2 * (1.0 - c_flat**2))
    v_wat_flat_z = r * v_air_z + factor_flat
    
    tau_flat = h_bot / v_wat_flat_z
    x_ref_bot = X + tau_flat * (r * v_air_x)
    y_ref_bot = Y + tau_flat * (r * v_air_y)
    
    # 8. DIC Displacement Calculation
    # The optical flow is the physical distance on the floor between the wavy ray hit and flat ray hit.
    dX = x_bot - x_ref_bot
    dY = y_bot - y_ref_bot
    
    # Grab true elevation purely for visual comparison in the dashboard
    eta_true, _, _ = get_harmonic_wave(X, Y, amp, wave_l, dir_deg, phase_rad)
    
    return X, Y, dX, dY, eta_true

# --- Execute Solver ---
X, Y, dX, dY, eta_true = generate_theoretical_flow(H_CAM, H_BOT, BOUNDS, int(GRID_SIZE), N_AIR, N_WATER, AMPLITUDE, WAVELENGTH, DIRECTION, PHASE)

# Pack into (N, N, 2) matrix for the CNN/Download
flow_matrix = np.stack([dX, dY], axis=-1)


# ==========================================
# 4. MAIN DASHBOARD TABS & PLOTTING
# ==========================================
tab1, tab2 = st.tabs(["Theoretical Generation & Download", "Upload & Validate Measurement"])

with tab1:
    st.subheader("Theoretical Displacement Fields")
    
    buffer = io.BytesIO()
    np.save(buffer, flow_matrix)
    st.download_button(
        label="💾 Download Theoretical Matrix (.npy)",
        data=buffer.getvalue(),
        file_name=f"theoretical_dic_{GRID_SIZE}x{GRID_SIZE}x2.npy",
        mime="application/octet-stream"
    )
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(eta_true, extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], origin='lower', cmap='ocean')
    ax1.set_title("True Wave Elevation [m]")
    fig.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(2, 2, 2)
    step = max(1, GRID_SIZE // 32)
    X_sub, Y_sub = X[::step, ::step], Y[::step, ::step]
    dX_sub, dY_sub = dX[::step, ::step], dY[::step, ::step]
    mag_sub = np.sqrt(dX_sub**2 + dY_sub**2)
    q = ax2.quiver(X_sub, Y_sub, dX_sub, dY_sub, mag_sub, cmap='coolwarm', scale_units='xy', scale=1.0)
    ax2.set_title("Displacement Vector Field (True Scale)")
    ax2.set_xlim([-BOUNDS, BOUNDS]); ax2.set_ylim([-BOUNDS, BOUNDS]); ax2.set_aspect('equal')
    fig.colorbar(q, ax=ax2)
    
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(dX, extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], origin='lower', cmap='coolwarm')
    ax3.set_title("Theoretical dX Matrix [m]")
    fig.colorbar(im3, ax=ax3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(dY, extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], origin='lower', cmap='coolwarm')
    ax4.set_title("Theoretical dY Matrix [m]")
    fig.colorbar(im4, ax=ax4)
    
    st.pyplot(fig)

with tab2:
    st.subheader("Validate Experimental Data")
    st.markdown(f"Upload measured displacement matrix. Must be a `.npy` file of shape **({GRID_SIZE}, {GRID_SIZE}, 2)** representing `[dX, dY]` in meters.")
    
    uploaded_file = st.file_uploader("Upload Measured DIC Matrix (.npy)", type=["npy"])
    
    if uploaded_file is not None:
        measured_matrix = np.load(uploaded_file)
        
        if measured_matrix.shape != (GRID_SIZE, GRID_SIZE, 2):
            st.error(f"Shape Mismatch! Expected ({GRID_SIZE}, {GRID_SIZE}, 2), but got {measured_matrix.shape}")
        else:
            st.success("File loaded successfully. Calculating Residuals (Measured - Theoretical)...")
            meas_dX = measured_matrix[:, :, 0]
            meas_dY = measured_matrix[:, :, 1]
            
            res_dX = meas_dX - dX
            res_dY = meas_dY - dY
            
            # --- 1. QUANTITATIVE METRICS ---
            st.markdown("### 1. Quantitative Error Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            rmse_x = np.sqrt(np.mean(res_dX**2))
            max_err_x = np.max(np.abs(res_dX))
            rmse_y = np.sqrt(np.mean(res_dY**2))
            max_err_y = np.max(np.abs(res_dY))
            
            col1.metric("RMSE dX", f"{rmse_x*1000:.2f} mm")
            col2.metric("Max Error dX", f"{max_err_x*1000:.2f} mm")
            col3.metric("RMSE dY", f"{rmse_y*1000:.2f} mm")
            col4.metric("Max Error dY", f"{max_err_y*1000:.2f} mm")
            
            # --- 2. 1D CROSS-SECTION TRANSECTS ---
            st.markdown("### 2. 1D Cross-Section (Transects)")
            st.info("**Phase Shift Check:** If the dashed measured line is shifted left/right from the solid theoretical line, the camera/wave phase is out of sync. If it's taller/shorter, the amplitude or scaling is off.")
            
            fig_1d, ax_1d = plt.subplots(1, 2, figsize=(14, 4))
            mid = int(GRID_SIZE // 2)
            
            # Transect for dX along X-axis (taking middle Y row)
            ax_1d[0].plot(X[mid, :], dX[mid, :], label="Theoretical dX", color='black', linewidth=2)
            ax_1d[0].plot(X[mid, :], meas_dX[mid, :], label="Measured dX", color='red', linestyle='--', linewidth=2)
            ax_1d[0].set_title(f"X-Axis Transect (dX at Y={Y[mid, 0]:.1f}m)")
            ax_1d[0].set_xlabel("X coordinate [m]")
            ax_1d[0].set_ylabel("dX Displacement [m]")
            ax_1d[0].legend()
            ax_1d[0].grid(True, linestyle=':', alpha=0.7)
            
            # Transect for dY along Y-axis (taking middle X column)
            ax_1d[1].plot(Y[:, mid], dY[:, mid], label="Theoretical dY", color='black', linewidth=2)
            ax_1d[1].plot(Y[:, mid], meas_dY[:, mid], label="Measured dY", color='blue', linestyle='--', linewidth=2)
            ax_1d[1].set_title(f"Y-Axis Transect (dY at X={X[0, mid]:.1f}m)")
            ax_1d[1].set_xlabel("Y coordinate [m]")
            ax_1d[1].set_ylabel("dY Displacement [m]")
            ax_1d[1].legend()
            ax_1d[1].grid(True, linestyle=':', alpha=0.7)
            
            st.pyplot(fig_1d)

            # --- 3. 2D RESIDUAL HEATMAPS ---
            st.markdown("### 3. 2D Residual Heatmaps")
            fig_res, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            im_rx = axes[0].imshow(res_dX, extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], origin='lower', cmap='bwr')
            axes[0].set_title("Residual dX (Error) [m]\n(Red=Measured is higher, Blue=Measured is lower)")
            fig_res.colorbar(im_rx, ax=axes[0])
            
            im_ry = axes[1].imshow(res_dY, extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], origin='lower', cmap='bwr')
            axes[1].set_title("Residual dY (Error) [m]")
            fig_res.colorbar(im_ry, ax=axes[1])
            
            st.pyplot(fig_res)
            st.info("Validation Check: If the experimental setup is perfectly calibrated, the residual plots above should be mostly white (values close to zero).")
