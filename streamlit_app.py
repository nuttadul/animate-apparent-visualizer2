
# streamlit_app_visualizer_anim_ui.py
# Full UI + smooth play/scrub (Streamlit-driven) with PERSISTENT CAMERA (streamlit-plotly-events).
# pip install streamlit numpy plotly streamlit-plotly-events

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

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

# ------------------ Geometry helpers ------------------
def cylinder_between(z0, z1, radius=0.3, n_theta=40, n_z=22):
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

def bone_traces(R_dist, show_bone=True, show_stick=True, show_xyproj=True, limit=3.0):
    # Build proximal+distal cylinders, anterior stripes, stick vectors, and XY floor + projections.
    traces = []

    # Proximal cylinder (fixed)
    xp, yp, zp = cylinder_between(0.0, 1.5, 0.3)
    traces.append(go.Surface(x=xp, y=yp, z=zp, showscale=False,
                             colorscale=[[0, "#d9e4ff"],[1, "#d9e4ff"]],
                             visible=show_bone, name="prox"))
    # Prox stripe (anterior, along +Y at theta = pi/2)
    sp = anterior_stripe_lines(0.0, 1.5, 0.3)
    traces.append(go.Scatter3d(x=sp[:,0], y=sp[:,1], z=sp[:,2], mode="lines",
                               line=dict(color="crimson", width=10), visible=show_bone, name="prox_stripe"))

    # Distal cylinder (rotated by R_dist)
    xd, yd, zd = cylinder_between(0.0, -1.5, 0.3)
    pts_d = np.stack([xd.flatten(), yd.flatten(), zd.flatten()], axis=1)
    td = transform_points(R_dist, pts_d)
    xd2 = td[:,0].reshape(xd.shape); yd2 = td[:,1].reshape(yd.shape); zd2 = td[:,2].reshape(zd.shape)
    traces.append(go.Surface(x=xd2, y=yd2, z=zd2, showscale=False,
                             colorscale=[[0, "#ffe0cc"],[1, "#ffe0cc"]],
                             visible=show_bone, name="dist"))
    # Dist stripe (rotated)
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
        arrow(origin, Ap*1.2, "seagreen", "A_prox"),
        arrow(origin, Ad*1.2, "crimson",  "A_dist"),
    ]

    # XY floor + projections of anterior directions
    rng=np.linspace(-limit, limit, 2); Xf,Yf=np.meshgrid(rng, rng); Zf=np.full_like(Xf,-limit)
    traces.append(go.Surface(x=Xf, y=Yf, z=Zf, opacity=0.10, showscale=False, visible=show_xyproj, name="floor"))
    zproj = -limit + 1e-3
    traces.append(go.Scatter3d(x=[0, Ap[0]*1.2], y=[0, Ap[1]*1.2], z=[zproj, zproj],
                               mode="lines", line=dict(color="rgba(46,139,87,0.65)", width=6),
                               visible=show_xyproj, name="Aprox→XY"))
    traces.append(go.Scatter3d(x=[0, Ad[0]*1.2], y=[0, Ad[1]*1.2], z=[zproj, zproj],
                               mode="lines", line=dict(color="rgba(220,20,60,0.75)", width=6),
                               visible=show_xyproj, name="Adist→XY"))
    return traces

def staged_rotation(alpha, beta, gamma, t):
    # Body-fixed staged rotation toward R = Ry(beta) Rx(alpha) Rz(gamma).
    if t <= 0: return np.eye(3)
    if t < 1.0:      # Torsion only
        return rot_z(gamma * t)
    elif t < 2.0:    # + Sagittal
        return rot_x(alpha * (t-1.0)) @ rot_z(gamma)
    else:            # + Coronal
        return rot_y(beta * (t-2.0)) @ rot_x(alpha) @ rot_z(gamma)

# ------------------ App Layout ------------------
st.set_page_config(page_title="Apparent Torsion Visualizer", layout="wide")

left, right = st.columns([1.65, 0.35], gap="large")

with left:
    st.markdown("<h1 style='margin-bottom:0'>Apparent Torsion Visualizer</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;margin-top:2px;margin-bottom:16px'>"
                "See the difference between true twist and what you see on axial CT slices."
                "</div>", unsafe_allow_html=True)

# --------- Right: compact inputs & toggles ---------
with right:
    st.markdown("### Inputs")
    alpha = st.number_input("Sagittal (about X) [deg]", -90, 90, 0, step=1, format="%d")
    beta  = st.number_input("Coronal (about Y) [deg]",  -90, 90, 0, step=1, format="%d")
    gamma = st.number_input("Torsion (about Z) [deg]", -180, 180, 0, step=1, format="%d")

    st.markdown("---")
    show_bone  = st.checkbox("Show bone", value=True)
    show_stick = st.checkbox("Show stick", value=True)
    show_xyproj = st.checkbox("Show XY floor", value=True)

    st.markdown("---")
    # Animation controls (Streamlit-driven, no Plotly frames -> no camera reset)
    if "t_progress" not in st.session_state: st.session_state["t_progress"] = 0.0
    if "playing"    not in st.session_state: st.session_state["playing"] = False
    if "cam"        not in st.session_state: st.session_state["cam"] = dict(eye=dict(x=1.4,y=1.4,z=1.2))

    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("⏵ Play" if not st.session_state["playing"] else "⏸ Pause"):
            st.session_state["playing"] = not st.session_state["playing"]
    with cols[1]:
        if st.button("⟲ Reset camera"):
            st.session_state["cam"] = dict(eye=dict(x=1.4,y=1.4,z=1.2))
    with cols[2]:
        if st.button("↺ Reset progress"):
            st.session_state["t_progress"] = 0.0

    st.write("")
    t = st.slider("Progress  (torsion → sagittal → coronal)", 0.0, 3.0, float(st.session_state["t_progress"]), 0.01)

# Auto-advance when playing (keeps camera; we re-apply saved camera each run)
if st.session_state.get("playing", False):
    st.session_state["t_progress"] = (st.session_state["t_progress"] + 0.03) % 3.0001
    st.experimental_rerun()  # smooth-ish play without touching camera

# ------------------ Compute & Show Result ------------------
R_final = staged_rotation(alpha, beta, gamma, 3.0)
A_prox = np.array([0,1,0]); A_dist_final = normalize(R_final @ A_prox)
phi_final = angle_xy(A_prox, A_dist_final)

with left:
    st.markdown("<div style='font-size:22px;color:#333;margin-top:4px'>Apparent Torsion (Final)</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:44px;font-weight:700;margin-top:-6px;margin-bottom:8px'>{phi_final:.1f}°</div>", unsafe_allow_html=True)

# ------------------ Build current frame from t ------------------
R_t = staged_rotation(alpha, beta, gamma, t)
limit=3.0
fig = go.Figure(data=bone_traces(R_t, show_bone, show_stick, show_xyproj, limit=limit))

# Restore camera from state and keep it sticky
fig.update_layout(scene=dict(xaxis=dict(range=[-limit,limit], title='X'),
                             yaxis=dict(range=[-limit,limit], title='Y'),
                             zaxis=dict(range=[-limit,limit], title='Z'),
                             aspectmode="cube",
                             bgcolor="rgb(248,248,248)"),
                  paper_bgcolor="white",
                  margin=dict(l=0,r=0,t=10,b=0),
                  height=780,
                  uirevision="sticky")

# Apply last camera if available
if st.session_state.get("cam"):
    fig.update_layout(scene_camera=st.session_state["cam"])

# Render via plotly_events so we can capture camera changes
events = plotly_events(fig, select_event=False, click_event=False, hover_event=False,
                       override_height=780, override_width="100%")

# Save camera when user moves it
if events and isinstance(events, list):
    for ev in events:
        rel = ev.get("relayout", {})
        # Plotly relayout keys include 'scene.camera', 'scene.camera.eye', etc.
        if any(k.startswith("scene.camera") for k in rel.keys()):
            # Merge partial updates
            prev = st.session_state.get("cam", dict(eye=dict(x=1.4,y=1.4,z=1.2)))
            eye    = rel.get("scene.camera.eye",    prev.get("eye"))
            center = rel.get("scene.camera.center", prev.get("center", {"x":0,"y":0,"z":0}))
            up     = rel.get("scene.camera.up",     prev.get("up", {"x":0,"y":0,"z":1}))
            st.session_state["cam"] = {"eye": eye, "center": center, "up": up}
            break
