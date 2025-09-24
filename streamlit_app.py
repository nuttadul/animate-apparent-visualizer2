
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ------------------ Math helpers ------------------
def deg2rad(d): return np.deg2rad(d)
def rad2deg(r): return np.rad2deg(r)
def rot_x(a):
    t=deg2rad(a); c,s=np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(b):
    t=deg2rad(b); c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(g):
    t=deg2rad(g); c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def normalize(v):
    n=np.linalg.norm(v); 
    return v if n == 0 else v/n
def angle_xy(u, v):
    u2 = np.array([u[0], u[1]]); v2 = np.array([v[0], v[1]])
    if np.linalg.norm(u2) < 1e-9 or np.linalg.norm(v2) < 1e-9: return float('nan')
    u2 = u2/np.linalg.norm(u2); v2 = v2/np.linalg.norm(v2)
    dot = float(np.clip(u2 @ v2, -1.0, 1.0))
    det = float(u2[0]*v2[1] - u2[1]*v2[0])
    return rad2deg(np.arctan2(det, dot))

# ------------------ Geometry & traces ------------------
def cylinder_between(z0, z1, radius=0.3, n_theta=40, n_z=20):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    zs = np.linspace(z0, z1, n_z)
    T, Z = np.meshgrid(thetas, zs)
    x = radius * np.cos(T); y = radius * np.sin(T); z = Z
    return x, y, z
def transform_points(R, pts):
    return (R @ pts.T).T
def anterior_stripe_lines(z0, z1, radius=0.3, theta=np.pi/2, n=60):
    zs = np.linspace(z0, z1, n)
    xs = radius*np.cos(theta)*np.ones_like(zs)
    ys = radius*np.sin(theta)*np.ones_like(zs)
    return np.stack([xs, ys, zs], axis=1)

def build_traces(R_dist, show_bone=True, show_stick=True, show_xyproj=True, limit=3.0):
    traces = []
    # Proximal cylinder (fixed)
    xp, yp, zp = cylinder_between(0.0, 1.5, 0.3)
    traces.append(go.Surface(x=xp, y=yp, z=zp, showscale=False,
                             colorscale=[[0, "#d9e4ff"],[1, "#d9e4ff"]],
                             visible=show_bone, name="prox"))
    # Prox stripe
    sp = anterior_stripe_lines(0.0, 1.5, 0.3)
    traces.append(go.Scatter3d(x=sp[:,0], y=sp[:,1], z=sp[:,2], mode="lines",
                               line=dict(color="crimson", width=10), visible=show_bone, name="prox_stripe"))
    # Distal cylinder (rotated)
    xd, yd, zd = cylinder_between(0.0, -1.5, 0.3)
    pts_d = np.stack([xd.flatten(), yd.flatten(), zd.flatten()], axis=1)
    td = transform_points(R_dist, pts_d)
    xd2 = td[:,0].reshape(xd.shape); yd2 = td[:,1].reshape(yd.shape); zd2 = td[:,2].reshape(zd.shape)
    traces.append(go.Surface(x=xd2, y=yd2, z=zd2, showscale=False,
                             colorscale=[[0, "#ffe0cc"],[1, "#ffe0cc"]],
                             visible=show_bone, name="dist"))
    # Dist stripe
    sd = anterior_stripe_lines(0.0, -1.5, 0.3)
    tsd = transform_points(R_dist, sd)
    traces.append(go.Scatter3d(x=tsd[:,0], y=tsd[:,1], z=tsd[:,2], mode="lines",
                               line=dict(color="crimson", width=10), visible=show_bone, name="dist_stripe"))
    # Stick vectors
    origin = np.zeros(3); Zp = np.array([0,0,1]); Ap = np.array([0,1,0])
    Zd = normalize(R_dist @ np.array([0,0,-1])); Ad = normalize(R_dist @ np.array([0,1,0]))
    def arrow(start, vec, color, name):
        s = np.array(start); v = np.array(vec); tip = s + v
        return go.Scatter3d(x=[s[0], tip[0]], y=[s[1], tip[1]], z=[s[2], tip[2]],
                            mode="lines", line=dict(color=color, width=12),
                            visible=show_stick, name=name)
    traces += [
        arrow(origin, Zp, "royalblue",  "Z_prox"),
        arrow(origin, Zd, "darkorange", "Z_dist"),
        arrow(origin, Ap*1.1, "seagreen", "A_prox"),
        arrow(origin, Ad*1.1, "crimson",  "A_dist"),
    ]
    # XY floor + projections
    rng=np.linspace(-limit, limit, 2); Xf,Yf=np.meshgrid(rng, rng); Zf=np.full_like(Xf,-limit)
    traces.append(go.Surface(x=Xf, y=Yf, z=Zf, opacity=0.10, showscale=False, visible=show_xyproj, name="floor"))
    zproj = -limit + 1e-3
    traces.append(go.Scatter3d(x=[0, Ap[0]*1.1], y=[0, Ap[1]*1.1], z=[zproj, zproj],
                               mode="lines", line=dict(color="rgba(46,139,87,0.65)", width=6),
                               visible=show_xyproj, name="Aprox→XY"))
    traces.append(go.Scatter3d(x=[0, Ad[0]*1.1], y=[0, Ad[1]*1.1], z=[zproj, zproj],
                               mode="lines", line=dict(color="rgba(220,20,60,0.75)", width=6),
                               visible=show_xyproj, name="Adist→XY"))
    return traces

def staged_rotation(alpha, beta, gamma, t):
    if t <= 0: return np.eye(3)
    if t < 1.0:  # torsion build
        return rot_z(gamma * t)
    elif t < 2.0:  # add sagittal
        return rot_x(alpha * (t-1.0)) @ rot_z(gamma)
    else:  # add coronal
        return rot_y(beta * (t-2.0)) @ rot_x(alpha) @ rot_z(gamma)

# ------------------ App ------------------
st.set_page_config(page_title="Apparent Torsion Visualizer", layout="wide")
left, right = st.columns([1.6, 0.40])

with left:
    st.markdown("<h1 style='margin-bottom:0'>Apparent Torsion Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;margin-top:2px;margin-bottom:16px'>"
                "See the difference between true twist and what you see on axial CT slices."
                "</div>", unsafe_allow_html=True)

with right:
    st.write("")

# Inputs (compact, right column)
with right:
    st.markdown("### Inputs")
    alpha = st.number_input("Sagittal (about X) [deg]", min_value=-90, max_value=90, value=0, step=1, format="%d")
    beta  = st.number_input("Coronal (about Y) [deg]",  min_value=-90, max_value=90, value=0, step=1, format="%d")
    gamma = st.number_input("Torsion (about Z) [deg]",  min_value=-180, max_value=180, value=0, step=1, format="%d")
    st.markdown("---")
    show_bone  = st.checkbox("Show bone", value=True)
    show_stick = st.checkbox("Show stick", value=True)
    show_xyproj = st.checkbox("Show XY floor", value=True)

# Compute final apparent torsion
R_final = staged_rotation(alpha, beta, gamma, 3.0)
A_prox = np.array([0,1,0]); A_dist_final = normalize(R_final @ A_prox)
phi_final = angle_xy(A_prox, A_dist_final)

with left:
    # Big main result
    st.markdown(f"<div style='font-size:22px;color:#333;margin-top:4px'>Apparent Torsion (Final)</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:44px;font-weight:700;margin-top:-6px;margin-bottom:8px'>{phi_final:.1f}°</div>", unsafe_allow_html=True)

# Build figure + frames (large viewport)
limit=3.0
R0 = staged_rotation(alpha, beta, gamma, 0.0)
fig = go.Figure(data=build_traces(R0, show_bone, show_stick, show_xyproj, limit=limit))

n_frames = 90
frames = []
for i in range(n_frames+1):
    t = 3.0 * i / n_frames
    R = staged_rotation(alpha, beta, gamma, t)
    frames.append(go.Frame(data=build_traces(R, show_bone, show_stick, show_xyproj, limit=limit), name=f"{t:.3f}"))
fig.frames = frames

# Plotly slider & Play/Pause inside the figure
steps = []
for i in range(n_frames+1):
    t = 3.0 * i / n_frames
    steps.append(dict(method="animate",
                      args=[[f"{t:.3f}"], {"mode":"immediate",
                                           "frame":{"duration":0,"redraw":True},
                                           "transition":{"duration":0}}],
                      label=f"{t:.2f}"))
sliders = [dict(active=0, steps=steps, x=0.12, y=0.04, len=0.76,
                currentvalue=dict(prefix="Progress: ", suffix=" (0→3)", font=dict(size=14)))]

updatemenus = [dict(type="buttons", showactive=False, x=0.12, y=0.10,
                    buttons=[dict(label="Play", method="animate",
                                  args=[None, {"fromcurrent":True, "frame":{"duration":35,"redraw":True},
                                               "transition":{"duration":0}}]),
                             dict(label="Pause", method="animate",
                                  args=[[None], {"mode":"immediate",
                                                 "frame":{"duration":0,"redraw":False},
                                                 "transition":{"duration":0}}])])]

fig.update_layout(scene=dict(xaxis=dict(range=[-limit,limit], title='X'),
                             yaxis=dict(range=[-limit,limit], title='Y'),
                             zaxis=dict(range=[-limit,limit], title='Z'),
                             aspectmode="cube",
                             bgcolor="rgb(248,248,248)"),
                  paper_bgcolor="white",
                  margin=dict(l=0,r=0,t=20,b=0),
                  sliders=sliders, updatemenus=updatemenus,
                  height=780,
                  # Set once; avoid setting in frames
                  scene_camera=dict(eye=dict(x=1.4,y=1.4,z=1.2)),
                  # Preserve camera & avoid transition resets
                  uirevision="keep-view",
                  transition={"duration": 0},
                  frame={"duration": 35, "redraw": True},
                  )

with left:
    st.plotly_chart(fig, use_container_width=True)
