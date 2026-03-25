import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import io

# --- Plotly Imports ---
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# --- Streamlit Setup ---
st.set_page_config(page_title="Snell's Law Displacement Dashboard", layout="wide")
st.title("Validate Displacement")

with st.expander("Readme: "):
    st.markdown("""
    ### Coordinate System
    * **X, Y**: Horizontal plane representing the water surface and the tank/ocean bottom.
    * **Z**: Vertical axis representing height (positive) and depth (negative).
    * **Origin (0,0,0)**: Located at the mean water surface exactly beneath the camera.
    * **Camera Position**: `(0, 0, Camera Height)`
    * **Tank Bottom**: `Z = -Water Depth`
    
    ### Matrix Indexing
    The full displacement vector field evaluates a `20m x 20m` physical area using a `256 x 256` uniform grid.
    * `X_app` and `Y_app` are generated using `numpy.meshgrid` and represent the **Apparent Positions** (where the camera *thinks* the dots are, assuming light travels in perfectly straight lines).
    * `dX_grid` and `dY_grid` map 1:1 to this grid. For example, the vector `[dX_grid[i, j], dY_grid[i, j]]` is the wave-induced optical displacement for the apparent point located at `(X_app[i, j], Y_app[i, j])`.

    The uploaded `.npz` file must contain separate 2D matrices (shape `256x256`) saved under the keys `'dX'` and `'dY'`. The data must follow a standard Cartesian layout (where index 0,0 is the minimum X and Y). For a 20x20m bounds area:
    * **`dX[0, 0]`** = Bottom-Left corner `(X = -10m, Y = -10m)`
    * **`dX[255, 0]`** = Top-Left corner `(X = -10m, Y = +10m)`
    * **`dX[0, 255]`** = Bottom-Right corner `(X = +10m, Y = -10m)`
    * **`dX[255, 255]`** = Top-Right corner `(X = +10m, Y = +10m)`
      
    ### Displacement (dX, dY)
    * $dX$ and $dY$ measure the **apparent optical shift** of a physical speckle on the tank floor.
    * **Calculation:** $dX = X_{flat} - X_{wave}$. The reference image viewed through flat water minus the deformed image viewed through waves (Green-Red).
    
    ### Physical Units
    * All spatial inputs and outputs (X, Y, dX, dY, Elevation) are in **METERS**.
    """)
                
# --- Sidebar Controls ---
st.sidebar.header("1. Physical Parameters")
h_camera = st.sidebar.number_input("Camera Height (m)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
d_water = st.sidebar.number_input("Water Depth (m)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
n_water = st.sidebar.number_input("Refractive Index (Water)", value=1.333, step=0.01)
n_air = 1.0

st.sidebar.header("2. Wave Parameters")
wave_len = st.sidebar.number_input("Wavelength (m)", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
wave_steepness = st.sidebar.number_input("Wave Steepness", min_value=0.0, max_value=0.15, value=0.08, step=0.01)
wave_dir_deg = st.sidebar.number_input("Wave Direction (deg)", min_value=0, max_value=360, value=45, step=5)

st.sidebar.header("3. Single Point Analysis")
x_app = st.sidebar.number_input("Apparent X (m)", min_value=-10.0, max_value=10.0, value=8.0, step=0.5)
y_app = st.sidebar.number_input("Apparent Y (m)", min_value=-10.0, max_value=10.0, value=-6.0, step=0.5)

st.sidebar.header("4. Experiment Data")
uploaded_file = st.sidebar.file_uploader("Upload measurement (.npz)", type=["npz"])

# --- Derived Variables ---
wave_amp = (wave_steepness * wave_len) / 2.0
wave_dir = np.deg2rad(wave_dir_deg)
k = 2 * np.pi / wave_len
kx = k * np.cos(wave_dir)
ky = k * np.sin(wave_dir)

# --- Computations ---

@st.cache_data
def compute_vector_field(h_cam, d_w, n_w, w_amp, k_x, k_y, grid_size=20.0, N_grid=256):
    """Computes the 256x256 displacement field matrix efficiently using vectorization."""
    x_lin = np.linspace(-grid_size/2, grid_size/2, N_grid)
    y_lin = np.linspace(-grid_size/2, grid_size/2, N_grid)
    X_app, Y_app = np.meshgrid(x_lin, y_lin)

    T = np.full_like(X_app, h_cam / (h_cam + d_w))
    for _ in range(10):
        px, py = X_app * T, Y_app * T
        pz = h_cam - T * (h_cam + d_w)
        S = w_amp * np.cos(k_x * px + k_y * py)
        f = pz - S
        S_prime = -w_amp * np.sin(k_x * px + k_y * py) * (k_x * X_app + k_y * Y_app)
        f_prime = -(h_cam + d_w) - S_prime
        T -= f / f_prime

    X_s_grid, Y_s_grid = X_app * T, Y_app * T
    Z_s_grid = h_cam - T * (h_cam + d_w)

    S_dx = -w_amp * k_x * np.sin(k_x * X_s_grid + k_y * Y_s_grid)
    S_dy = -w_amp * k_y * np.sin(k_x * X_s_grid + k_y * Y_s_grid)

    Nx, Ny, Nz = -S_dx, -S_dy, np.ones_like(X_s_grid)
    N_norm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    nx, ny, nz = Nx/N_norm, Ny/N_norm, Nz/N_norm

    Vx, Vy, Vz = X_s_grid, Y_s_grid, Z_s_grid - h_cam
    V_norm = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    dx_in, dy_in, dz_in = Vx/V_norm, Vy/V_norm, Vz/V_norm

    r = n_air / n_w
    c_grid = -(nx*dx_in + ny*dy_in + nz*dz_in)
    coeff = r * c_grid - np.sqrt(1 - r**2 * (1 - c_grid**2))
    dx_out = r * dx_in + coeff * nx
    dy_out = r * dy_in + coeff * ny
    dz_out = r * dz_in + coeff * nz

    t_bot_grid = (-d_w - Z_s_grid) / dz_out
    X_true = X_s_grid + t_bot_grid * dx_out
    Y_true = Y_s_grid + t_bot_grid * dy_out

    R_true = np.sqrt(X_true**2 + Y_true**2)
    R_s = R_true * (h_cam / (h_cam + d_w))

    for _ in range(15):
        diff = R_true - R_s
        hyp_w = np.sqrt(diff**2 + d_w**2)
        hyp_a = np.sqrt(R_s**2 + h_cam**2)
        sin_w, sin_a = diff / hyp_w, R_s / hyp_a
        f = n_w * sin_w - n_air * sin_a
        dsin_w = -(d_w**2) / (hyp_w**3)
        dsin_a = (h_cam**2) / (hyp_a**3)
        f_prime = n_w * dsin_w - n_air * dsin_a
        R_s -= f / f_prime

    safe_R_true = np.where(R_true == 0, 1e-10, R_true)
    scale = R_s / safe_R_true
    X_sf, Y_sf = X_true * scale, Y_true * scale

    t_virt_grid = (h_cam + d_w) / h_cam
    X_virt = X_sf * t_virt_grid
    Y_virt = Y_sf * t_virt_grid

    dX_grid = X_virt - X_app
    dY_grid = Y_virt - Y_app
    return X_app, Y_app, dX_grid, dY_grid

def calc_single_point():
    cam_pos = np.array([0.0, 0.0, h_camera])
    app_dot = np.array([x_app, y_app, -d_water])

    def surface(x, y): return wave_amp * np.cos(kx * x + ky * y)
    def surface_dx(x, y): return -wave_amp * kx * np.sin(kx * x + ky * y)
    def surface_dy(x, y): return -wave_amp * ky * np.sin(kx * x + ky * y)

    def line_surface_intersection(t):
        px = cam_pos[0] + t * (app_dot[0] - cam_pos[0])
        py = cam_pos[1] + t * (app_dot[1] - cam_pos[1])
        pz = cam_pos[2] + t * (app_dot[2] - cam_pos[2])
        return pz - surface(px, py)

    t_guess = h_camera / (h_camera + d_water)
    t_hit = fsolve(line_surface_intersection, t_guess)[0]
    surf_pos = cam_pos + t_hit * (app_dot - cam_pos)
    x_s, y_s, z_s = surf_pos

    n_vec = np.array([-surface_dx(x_s, y_s), -surface_dy(x_s, y_s), 1.0])
    n_hat = n_vec / np.linalg.norm(n_vec)

    v_in = surf_pos - cam_pos
    d_in = v_in / np.linalg.norm(v_in)

    r = n_air / n_water
    c = -np.dot(n_hat, d_in)
    d_out = r * d_in + (r * c - np.sqrt(1 - r**2 * (1 - c**2))) * n_hat

    t_bottom = (-d_water - z_s) / d_out[2]
    true_dot = surf_pos + t_bottom * d_out

    def flat_interface_root(vars):
        x_sf, y_sf = vars
        S_f = np.array([x_sf, y_sf, 0.0])
        V_w = S_f - true_dot
        D_w = V_w / np.linalg.norm(V_w)
        V_a = cam_pos - S_f
        D_a = V_a / np.linalg.norm(V_a)
        eq1 = n_water * D_w[0] - n_air * D_a[0]
        eq2 = n_water * D_w[1] - n_air * D_a[1]
        return [eq1, eq2]

    t_guess_flat = -true_dot[2] / (cam_pos[2] - true_dot[2])
    S_guess = true_dot + t_guess_flat * (cam_pos - true_dot)
    x_sf, y_sf = fsolve(flat_interface_root, [S_guess[0], S_guess[1]])
    surf_pos_flat = np.array([x_sf, y_sf, 0.0])

    t_bottom_virt = (h_camera + d_water) / h_camera
    virt_flat_dot = cam_pos + t_bottom_virt * (surf_pos_flat - cam_pos)
    dr = virt_flat_dot - app_dot
    
    return cam_pos, app_dot, surf_pos, true_dot, surf_pos_flat, virt_flat_dot, n_hat, dr, surface

# --- Main App Execution ---

with st.spinner("Computing Physics..."):
    # 1. Compute Full Field
    grid_size = 20.0
    N_grid = 256
    X_app, Y_app, dX_grid, dY_grid = compute_vector_field(h_camera, d_water, n_water, wave_amp, kx, ky, grid_size, N_grid)
    dr_mag = np.sqrt(dX_grid**2 + dY_grid**2)
    
    # 2. Compute Single Point
    try:
        single_pt_data = calc_single_point()
        single_pt_success = True
    except Exception as e:
        single_pt_success = False
        st.error(f"Single point solver failed (likely ray missed surface due to extreme steepness). {e}")

# --- Plotly Helpers ---
def add_rays_2d_plotly(fig, dim1, dim2, c_pos, s_pos, a_pos, t_pos, n_h, s_pos_f, v_flat_pos, show_legend=False):
    c1, c2 = c_pos[dim1], c_pos[dim2]
    s1, s2 = s_pos[dim1], s_pos[dim2]
    a1, a2 = a_pos[dim1], a_pos[dim2]
    t1, t2 = t_pos[dim1], t_pos[dim2]
    sf1, sf2 = s_pos_f[dim1], s_pos_f[dim2]
    vf1, vf2 = v_flat_pos[dim1], v_flat_pos[dim2]
    nh1, nh2 = n_h[dim1], n_h[dim2]

    # Wavy Rays
    fig.add_trace(go.Scatter(x=[c1, s1], y=[c2, s2], mode='lines', line=dict(color='black', width=2), name='Air Ray (Wavy)', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[s1, a1], y=[s2, a2], mode='lines', line=dict(color='red', width=2, dash='dash'), name='Apparent Line (Wavy)', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[s1, t1], y=[s2, t2], mode='lines', line=dict(color='blue', width=2), name='True Ray (Wavy)', showlegend=show_legend))
    
    # Flat Rays
    fig.add_trace(go.Scatter(x=[c1, sf1], y=[c2, sf2], mode='lines', line=dict(color='green', width=2), opacity=0.6, name='Air Ray (Flat)', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[sf1, vf1], y=[sf2, vf2], mode='lines', line=dict(color='green', width=2, dash='dash'), opacity=0.6, name='Apparent Line (Flat)', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[t1, sf1], y=[t2, sf2], mode='lines', line=dict(color='green', width=2, dash='dot'), opacity=0.6, name='True Ray (Flat)', showlegend=show_legend))
    
    # Normal
    normal_len = 2.0
    fig.add_trace(go.Scatter(x=[s1 - nh1*normal_len, s1 + nh1*normal_len],
                             y=[s2 - nh2*normal_len, s2 + nh2*normal_len],
                             mode='lines', line=dict(color='gray', width=2, dash='dot'), showlegend=False))

    # Displacement Vector (dr) as an Annotation
    dr_len = np.linalg.norm(np.array([vf1-a1, vf2-a2]))
    if dr_len > 1e-4:
        fig.add_annotation(
            x=vf1, y=vf2, ax=a1, ay=a2, xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='magenta'
        )

    # Points
    fig.add_trace(go.Scatter(x=[c1], y=[c2], mode='markers', marker=dict(color='black', symbol='square', size=10), name='Camera', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[a1], y=[a2], mode='markers', marker=dict(color='red', symbol='x', size=10), name='Virtual Dot (Wave)', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[vf1], y=[vf2], mode='markers', marker=dict(color='green', symbol='diamond', size=10), name='Virtual Dot (Flat)', showlegend=show_legend))
    fig.add_trace(go.Scatter(x=[t1], y=[t2], mode='markers', marker=dict(color='blue', symbol='circle', size=10), name='Physical Dot', showlegend=show_legend))


# --- Render Tabs ---
tab1, tab2, tab3 = st.tabs(["Point Raytracing (3D)", "Vector Displacement Field", "Experiment Comparison"])

with tab1:
    if single_pt_success:
        cam_pos, app_dot, surf_pos, true_dot, surf_pos_flat, virt_flat_dot, n_hat, dr, surface = single_pt_data
        x_s, y_s, z_s = surf_pos 

        x_min = min(cam_pos[0], x_app, true_dot[0], virt_flat_dot[0]) - 2
        x_max = max(cam_pos[0], x_app, true_dot[0], virt_flat_dot[0]) + 2
        y_min = min(cam_pos[1], y_app, true_dot[1], virt_flat_dot[1]) - 2
        y_max = max(cam_pos[1], y_app, true_dot[1], virt_flat_dot[1]) + 2
        
        # --- 3D Plot ---
        st.subheader("3D View")
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        Z_surf = surface(X, Y)

        fig3d = go.Figure()
        # Water Surface
        fig3d.add_trace(go.Surface(x=X, y=Y, z=Z_surf, colorscale=[[0, 'cyan'], [1, 'cyan']], opacity=0.5, showscale=False))
        
        # Wavy Rays
        fig3d.add_trace(go.Scatter3d(x=[cam_pos[0], surf_pos[0]], y=[cam_pos[1], surf_pos[1]], z=[cam_pos[2], surf_pos[2]], mode='lines', line=dict(color='black', width=4), name='Air Ray'))
        fig3d.add_trace(go.Scatter3d(x=[surf_pos[0], app_dot[0]], y=[surf_pos[1], app_dot[1]], z=[surf_pos[2], app_dot[2]], mode='lines', line=dict(color='red', width=4, dash='dash'), name='Apparent Line'))
        fig3d.add_trace(go.Scatter3d(x=[surf_pos[0], true_dot[0]], y=[surf_pos[1], true_dot[1]], z=[surf_pos[2], true_dot[2]], mode='lines', line=dict(color='blue', width=4), name='True Ray'))
        
        # Flat Rays
        fig3d.add_trace(go.Scatter3d(x=[cam_pos[0], surf_pos_flat[0]], y=[cam_pos[1], surf_pos_flat[1]], z=[cam_pos[2], surf_pos_flat[2]], mode='lines', line=dict(color='green', width=2), name='Air Ray (Flat)'))
        fig3d.add_trace(go.Scatter3d(x=[surf_pos_flat[0], virt_flat_dot[0]], y=[surf_pos_flat[1], virt_flat_dot[1]], z=[surf_pos_flat[2], virt_flat_dot[2]], mode='lines', line=dict(color='green', width=2, dash='dash'), name='Apparent Line (Flat)'))
        fig3d.add_trace(go.Scatter3d(x=[true_dot[0], surf_pos_flat[0]], y=[true_dot[1], surf_pos_flat[1]], z=[true_dot[2], surf_pos_flat[2]], mode='lines', line=dict(color='green', width=2, dash='dot'), name='True Ray (Flat)'))

        # Displacement Vector 3D Line
        fig3d.add_trace(go.Scatter3d(x=[app_dot[0], virt_flat_dot[0]], y=[app_dot[1], virt_flat_dot[1]], z=[app_dot[2], virt_flat_dot[2]], mode='lines', line=dict(color='magenta', width=6), name='Displacement (dr)'))

        # Markers
        fig3d.add_trace(go.Scatter3d(x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]], mode='markers', marker=dict(color='black', symbol='square', size=6), name='Camera'))
        fig3d.add_trace(go.Scatter3d(x=[app_dot[0]], y=[app_dot[1]], z=[app_dot[2]], mode='markers', marker=dict(color='red', symbol='x', size=6), name='Virtual Dot (Wave)'))
        fig3d.add_trace(go.Scatter3d(x=[virt_flat_dot[0]], y=[virt_flat_dot[1]], z=[virt_flat_dot[2]], mode='markers', marker=dict(color='green', symbol='diamond', size=6), name='Virtual Dot (Flat)'))
        fig3d.add_trace(go.Scatter3d(x=[true_dot[0]], y=[true_dot[1]], z=[true_dot[2]], mode='markers', marker=dict(color='blue', symbol='circle', size=6), name='Physical Dot'))

        fig3d.update_layout(
            scene=dict(aspectmode='data', xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z Depth (m)'),
            height=700, margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # --- 2D Projections ---
        st.subheader("2D Axis Projections")
        col1, col2, col3 = st.columns(3)

        # Top View (X-Y)
        fig_top = go.Figure()
        fig_top.add_trace(go.Contour(x=np.linspace(x_min, x_max, 50), y=np.linspace(y_min, y_max, 50), z=Z_surf, colorscale='Blues', opacity=0.4, showscale=False))
        add_rays_2d_plotly(fig_top, 0, 1, cam_pos, surf_pos, app_dot, true_dot, n_hat, surf_pos_flat, virt_flat_dot, show_legend=True)
        fig_top.update_layout(title="Top View (X-Y)", xaxis_title="X", yaxis_title="Y", height=500, yaxis=dict(scaleanchor="x", scaleratio=1), showlegend=False)
        with col1: st.plotly_chart(fig_top, use_container_width=True)

        # Front View (X-Z)
        fig_front = go.Figure()
        x_line = np.linspace(x_min, x_max, 500)
        z_line = surface(x_line, y_s)
        fig_front.add_trace(go.Scatter(x=np.concatenate([x_line, x_line[::-1]]), y=np.concatenate([z_line, np.full_like(z_line, -d_water - 2)]), fill='toself', fillcolor='rgba(224, 247, 250, 0.5)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig_front.add_trace(go.Scatter(x=x_line, y=z_line, mode='lines', line=dict(color='#00838f', width=2), name=f'Wave Slice at Y={y_s:.1f}'))
        add_rays_2d_plotly(fig_front, 0, 2, cam_pos, surf_pos, app_dot, true_dot, n_hat, surf_pos_flat, virt_flat_dot)
        fig_front.update_layout(title="Front View (X-Z)", xaxis_title="X", yaxis_title="Z", height=500, yaxis=dict(scaleanchor="x", scaleratio=1), showlegend=False)
        with col2: st.plotly_chart(fig_front, use_container_width=True)

        # Side View (Y-Z)
        fig_side = go.Figure()
        y_line = np.linspace(y_min, y_max, 500)
        z_line = surface(x_s, y_line)
        fig_side.add_trace(go.Scatter(x=np.concatenate([y_line, y_line[::-1]]), y=np.concatenate([z_line, np.full_like(z_line, -d_water - 2)]), fill='toself', fillcolor='rgba(224, 247, 250, 0.5)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig_side.add_trace(go.Scatter(x=y_line, y=z_line, mode='lines', line=dict(color='#00838f', width=2), name=f'Wave Slice at X={x_s:.1f}'))
        add_rays_2d_plotly(fig_side, 1, 2, cam_pos, surf_pos, app_dot, true_dot, n_hat, surf_pos_flat, virt_flat_dot)
        fig_side.update_layout(title="Side View (Y-Z)", xaxis_title="Y", yaxis_title="Z", height=500, yaxis=dict(scaleanchor="x", scaleratio=1), showlegend=False)
        with col3: st.plotly_chart(fig_side, use_container_width=True)

with tab2:
    st.subheader(f"Theoretical Displacement Vector Field True Scale ({grid_size}x{grid_size}m, {N_grid}x{N_grid})")
    
    fig2 = go.Figure()
    
    # Heatmap
    x_1d = X_app[0,:]
    y_1d = Y_app[:,0]
    fig2.add_trace(go.Heatmap(x=x_1d, y=y_1d, z=dr_mag, colorscale='Viridis', colorbar=dict(title="Magnitude |dr| (m)")))

    # Quiver
    step = 12
    X_dec = X_app[::step, ::step].flatten()
    Y_dec = Y_app[::step, ::step].flatten()
    U_dec = dX_grid[::step, ::step].flatten()
    V_dec = dY_grid[::step, ::step].flatten()

    fig_q = ff.create_quiver(X_dec, Y_dec, U_dec, V_dec, scale=1, arrow_scale=0.15, line=dict(color='white', width=1))
    fig2.add_traces(fig_q.data)

    # Single Assessed Point Overlay
    if single_pt_success:
        fig2.add_trace(go.Scatter(x=[x_app], y=[y_app], mode='markers', marker=dict(color='red', symbol='x', size=12, line=dict(width=3)), name='Assessed Point'))
        fig2.add_annotation(
            x=app_dot[0] + dr[0], y=app_dot[1] + dr[1], ax=app_dot[0], ay=app_dot[1],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='red'
        )

    fig2.update_layout(
        xaxis_title="Apparent X Position (m)", yaxis_title="Apparent Y Position (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1), height=800
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Provide download button for the theoretical matrix
    buffer = io.BytesIO()
    np.savez(buffer, X=X_app, Y=Y_app, dX=dX_grid, dY=dY_grid)
    st.download_button(
        label="📥 Download Theoretical Displacement Matrix (.npz)",
        data=buffer.getvalue(),
        file_name="theoretical_displacement.npz",
        mime="application/octet-stream"
    )

with tab3:
    if uploaded_file is not None:
        try:
            exp_data = np.load(uploaded_file)
            exp_dX = exp_data['dX']
            exp_dY = exp_data['dY']
            
            if exp_dX.shape != dX_grid.shape:
                st.error(f"Shape mismatch! Uploaded data is {exp_dX.shape}, but theoretical model is {dX_grid.shape}. Please ensure they are evaluated on the exact same 256x256 meshgrid.")
            else:
                exp_dr_mag = np.sqrt(exp_dX**2 + exp_dY**2)
                err_X = dX_grid - exp_dX
                err_Y = dY_grid - exp_dY
                err_mag = np.sqrt(err_X**2 + err_Y**2)
                
                # ... (existing code just above this)
                rmse_x = np.sqrt(np.mean(err_X**2))
                rmse_y = np.sqrt(np.mean(err_Y**2))
                rmse_total = np.sqrt(np.mean(err_mag**2))
                
                # --- NEW: Calculate Percentage Accuracy ---
                # Using Global Relative Error (Normalized RMSE) to avoid division by zero
                rms_true_dr = np.sqrt(np.mean(dr_mag**2))
                
                # Prevent division by zero just in case the true field is perfectly flat
                global_relative_error = rmse_total / (rms_true_dr + 1e-10) 
                
                # Convert to percentage and clamp at 0% minimum
                percentage_accuracy = max(0.0, 100.0 * (1.0 - global_relative_error))
                
                # --- UPDATED: Display 4 columns instead of 3 ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE (X)", f"{rmse_x:.4f} m")
                col2.metric("RMSE (Y)", f"{rmse_y:.4f} m")
                col3.metric("Total RMSE", f"{rmse_total:.4f} m")
                
                # Display the accuracy, color it green if >95% (to match your 5% noise threshold)
                if percentage_accuracy >= 95.0:
                    col4.success(f"Accuracy: {percentage_accuracy:.1f}%")
                else:
                    col4.warning(f"Accuracy: {percentage_accuracy:.1f}%")
                
                vmax = max(np.max(dr_mag), np.max(exp_dr_mag))
                # ... (existing code continues below to make the subplots)

                fig3 = make_subplots(rows=1, cols=3, subplot_titles=("Theoretical Displacement", "Experimental Displacement", "Error Map (Theory - Exp)"))
                
                x_1d = X_app[0,:]
                y_1d = Y_app[:,0]
                
                # Plot 1: Theoretical
                fig3.add_trace(go.Heatmap(x=x_1d, y=y_1d, z=dr_mag, coloraxis="coloraxis"), row=1, col=1)
                q_theo = ff.create_quiver(X_dec, Y_dec, U_dec, V_dec, scale=1, arrow_scale=0.15, line=dict(color='white', width=1))
                fig3.add_trace(q_theo.data[0], row=1, col=1)

                # Plot 2: Experimental
                exp_U_dec = exp_dX[::step, ::step].flatten()
                exp_V_dec = exp_dY[::step, ::step].flatten()
                fig3.add_trace(go.Heatmap(x=x_1d, y=y_1d, z=exp_dr_mag, coloraxis="coloraxis"), row=1, col=2)
                q_exp = ff.create_quiver(X_dec, Y_dec, exp_U_dec, exp_V_dec, scale=1, arrow_scale=0.15, line=dict(color='white', width=1))
                fig3.add_trace(q_exp.data[0], row=1, col=2)

                # Plot 3: Error
                err_U_dec = err_X[::step, ::step].flatten()
                err_V_dec = err_Y[::step, ::step].flatten()
                fig3.add_trace(go.Heatmap(x=x_1d, y=y_1d, z=err_mag, coloraxis="coloraxis2"), row=1, col=3)
                q_err = ff.create_quiver(X_dec, Y_dec, err_U_dec, err_V_dec, scale=1, arrow_scale=0.15, line=dict(color='white', width=1))
                fig3.add_trace(q_err.data[0], row=1, col=3)

                fig3.update_layout(
                    coloraxis=dict(colorscale='Viridis', cmin=0, cmax=vmax, colorbar=dict(title='Mag (m)', x=0.63)),
                    coloraxis2=dict(colorscale='Inferno', cmin=0, colorbar=dict(title='Error (m)', x=1.0)),
                    height=600, showlegend=False
                )
                fig3.update_xaxes(title_text="X Apparent (m)")
                fig3.update_yaxes(title_text="Y Apparent (m)", scaleanchor="x", scaleratio=1)
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # --- 1D Cross Sections ---
                st.subheader(f"1D Cross-Sections at Assessed Point (X={x_app}m, Y={y_app}m)")
                
                # Find closest indices in the meshgrid to the assessed point
                idx_x = np.argmin(np.abs(x_1d - x_app))
                idx_y = np.argmin(np.abs(y_1d - y_app))
                
                fig4 = make_subplots(rows=1, cols=2, 
                                     subplot_titles=(f"Slice along X-axis (at Y ≈ {y_1d[idx_y]:.2f}m)", 
                                                     f"Slice along Y-axis (at X ≈ {x_1d[idx_x]:.2f}m)"))
                
                # Cross-section along X
                fig4.add_trace(go.Scatter(x=x_1d, y=dr_mag[idx_y, :], mode='lines', name='Theory |dr|', line=dict(color='blue', width=2)), row=1, col=1)
                fig4.add_trace(go.Scatter(x=x_1d, y=exp_dr_mag[idx_y, :], mode='lines+markers', name='Experiment |dr|', line=dict(color='orange', width=2, dash='dot'), marker=dict(size=4)), row=1, col=1)
                fig4.add_vline(x=x_app, line_dash="dash", line_color="red", annotation_text="Assessed X", row=1, col=1)

                # Cross-section along Y
                fig4.add_trace(go.Scatter(x=y_1d, y=dr_mag[:, idx_x], mode='lines', name='Theory |dr|', showlegend=False, line=dict(color='blue', width=2)), row=1, col=2)
                fig4.add_trace(go.Scatter(x=y_1d, y=exp_dr_mag[:, idx_x], mode='lines+markers', name='Experiment |dr|', showlegend=False, line=dict(color='orange', width=2, dash='dot'), marker=dict(size=4)), row=1, col=2)
                fig4.add_vline(x=y_app, line_dash="dash", line_color="red", annotation_text="Assessed Y", row=1, col=2)

                fig4.update_xaxes(title_text="X Apparent (m)", row=1, col=1)
                fig4.update_yaxes(title_text="Displacement Magnitude |dr| (m)", row=1, col=1)
                fig4.update_xaxes(title_text="Y Apparent (m)", row=1, col=2)
                fig4.update_yaxes(title_text="Displacement Magnitude |dr| (m)", row=1, col=2)
                
                fig4.update_layout(height=400, hovermode="x unified", margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig4, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing the uploaded .npz file: {e}")
    else:
        st.info("Upload the experimental data (.npz) via the sidebar to view comparison metrics.")
        
        if st.button("Generate & Download Mock Experimental Data"):
            noise_x = np.random.normal(0, 0.02, dX_grid.shape) + 0.03 * np.sin(X_app)
            noise_y = np.random.normal(0, 0.02, dY_grid.shape) + 0.03 * np.cos(Y_app)
            mock_dx = dX_grid + noise_x
            mock_dy = dY_grid + noise_y
            
            mock_buffer = io.BytesIO()
            np.savez(mock_buffer, X=X_app, Y=Y_app, dX=mock_dx, dY=mock_dy)
            
            st.download_button(
                label="📥 Download 'dic_experiment.npz' (Mock Data)",
                data=mock_buffer.getvalue(),
                file_name="dic_experiment.npz",
                mime="application/octet-stream"
            )
            st.success("Mock data generated! Download it and re-upload it via the sidebar.")
