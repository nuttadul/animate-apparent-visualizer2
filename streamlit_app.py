
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# =========================
# Math helpers
# =========================
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
    u2 = np.array([u[0], u[1]])
    v2 = np.array([v[0], v[1]])
    if np.linalg.norm(u2) < 1e-9 or np.linalg.norm(v2) < 1e-9:
        return float('nan')
    u2 = u2/np.linalg.norm(u2); v2 = v2/np.linalg.norm(v2)
    dot = float(np.clip(u2 @ v2, -1.0, 1.0))
    det = float(u2[0]*v2[1] - u2[1]*v2[0])
    return rad2deg(np.arctan2(det, dot))

# =========================
# Geometry helpers (two-segment bone with anterior stripe)
# =========================
def cylinder_between(z0, z1, radius=0.2, n_theta=50, n_z=20):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    zs = np.linspace(z0, z1, n_z)
    T, Z = np.meshgrid(thetas, zs)
    x = radius * np.cos(T)
    y = radius * np.sin(T)
    z = Z
    return x, y, z

def transform_points(R, pts):
    return (R @ pts.T).T

def anterior_stripe_lines(z0, z1, radius=0.2, theta=np.pi/2, n=60):
    zs = np.linspace(z0, z1, n)
    xs = radius*np.cos(theta)*np.ones_like(zs)
    ys = radius*np.sin(theta)*np.ones_like(zs)
    return np.stack([xs, ys, zs], axis=1)

def bone_pair_traces(R_dist, L_prox=0.9, L_dist=0.9, radius=0.2,
                     color_prox='#d9e4ff', color_dist='#ffe0cc',
                     stripe_color='crimson', edge_color='#888'):
    traces = []
    # Proximal: fixed along +Z from 0..L_prox
    xp, yp, zp = cylinder_between(0.0, L_prox, radius)
    tp = np.stack([xp.flatten(), yp.flatten(), zp.flatten()], axis=1)
    xp2 = tp[:,0].reshape(xp.shape); yp2 = tp[:,1].reshape(yp.shape); zp2 = tp[:,2].reshape(zp.shape)
    surf_p = go.Surface(x=xp2, y=yp2, z=zp2, showscale=False,
                        colorscale=[[0, color_prox],[1, color_prox]], name='Prox bone', opacity=1.0)
    traces.append(surf_p)
    sp = anterior_stripe_lines(0.0, L_prox, radius)
    tsp = sp
    traces.append(go.Scatter3d(x=tsp[:,0], y=tsp[:,1], z=tsp[:,2], mode='lines',
                               line=dict(color=stripe_color, width=8), name='Anterior stripe (prox)', showlegend=True))
    # Distal: along -Z from 0..-L_dist then rotated
    xd, yd, zd = cylinder_between(0.0, -L_dist, radius)
    pts_d = np.stack([xd.flatten(), yd.flatten(), zd.flatten()], axis=1)
    td = transform_points(R_dist, pts_d)
    xd2 = td[:,0].reshape(xd.shape); yd2 = td[:,1].reshape(yd.shape); zd2 = td[:,2].reshape(zd.shape)
    surf_d = go.Surface(x=xd2, y=yd2, z=zd2, showscale=False,
                        colorscale=[[0, color_dist],[1, color_dist]], name='Dist bone', opacity=1.0)
    traces.append(surf_d)
    sd = anterior_stripe_lines(0.0, -L_dist, radius)
    tsd = transform_points(R_dist, sd)
    traces.append(go.Scatter3d(x=tsd[:,0], y=tsd[:,1], z=tsd[:,2], mode='lines',
                               line=dict(color=stripe_color, width=8), name='Anterior stripe (dist)', showlegend=False))
    return traces

def axes_traces(limit=1.6):
    x=go.Scatter3d(x=[0,limit], y=[0,0], z=[0,0], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False, name='X')
    y=go.Scatter3d(x=[0,0], y=[0,limit], z=[0,0], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False, name='Y')
    z=go.Scatter3d(x=[0,0], y=[0,0], z=[0,limit], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False, name='Z')
    return [x,y,z]

def xy_floor(limit=1.6, opacity=0.10):
    rng=np.linspace(-limit, limit, 2)
    X,Y=np.meshgrid(rng, rng)
    return go.Surface(x=X,y=Y,z=np.full_like(X,-limit),opacity=opacity,showscale=False,name='XY floor')

def projected_line_to_xy(vec, color='rgba(0,0,0,0.7)', width=6, limit=1.6):
    s=np.array([0,0,0]); v=np.array(vec); tip=s+v
    z = -limit + 1e-3
    return go.Scatter3d(x=[s[0], tip[0]], y=[s[1], tip[1]], z=[z, z],
                        mode='lines', line=dict(color=color, width=width),
                        showlegend=False, name='proj→XY')

def build_scene(R_dist, show_bone=True, show_xyproj=True, title=""):
    limit=1.8
    traces = []
    traces += axes_traces(limit)
    if show_bone:
        traces += bone_pair_traces(R_dist)
    if show_xyproj:
        traces.append(xy_floor(limit, opacity=0.12))
        Ad = normalize(R_dist @ np.array([0,1,0]))
        traces.append(projected_line_to_xy(np.array([0,1,0])*0.9, color='rgba(46,139,87,0.65)', limit=limit))
        traces.append(projected_line_to_xy(Ad*0.9, color='rgba(220,20,60,0.75)', limit=limit))
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene=dict(
        xaxis=dict(title='X', range=[-limit,limit]),
        yaxis=dict(title='Y', range=[-limit,limit]),
        zaxis=dict(title='Z', range=[-limit,limit]),
        aspectmode='cube',
        bgcolor='rgb(248,248,248)'
    ), paper_bgcolor='white', margin=dict(l=0,r=0,t=30,b=0))
    return fig

# =========================
# Step build-up
# =========================
def staged_rotation(alpha, beta, gamma, progress):
    if progress <= 0: return np.eye(3)
    if progress < 1.0:
        t1 = progress
        return rot_z(gamma * t1)
    elif progress < 2.0:
        t2 = progress - 1.0
        return rot_x(alpha * t2) @ rot_z(gamma)
    else:
        t3 = progress - 2.0
        return rot_y(beta * t3) @ rot_x(alpha) @ rot_z(gamma)

# =========================
# App
# =========================
st.set_page_config(page_title="Apparent Torsion Visualizer – Step Build", layout="wide")
st.title("Apparent Torsion Visualizer – Step Build")
st.caption("Set target angles and use the Progress slider to add torsion, then sagittal, then coronal — step by step.")

if "targets" not in st.session_state:
    st.session_state["targets"] = dict(alpha=0.0, beta=0.0, gamma=0.0)

left, right = st.columns([1.4, 1.0])

with right:
    with st.form("targets_form"):
        alpha_in = st.slider("Sagittal (about X) [deg]", -60.0, 60.0, float(st.session_state["targets"]["alpha"]), 0.5)
        beta_in  = st.slider("Coronal (about Y) [deg]", -60.0, 60.0, float(st.session_state["targets"]["beta"]), 0.5)
        gamma_in = st.slider("Torsion (about Z) [deg]", -90.0, 90.0, float(st.session_state["targets"]["gamma"]), 0.5)
        applied = st.form_submit_button("Apply targets")
        if applied:
            st.session_state["targets"].update(alpha=alpha_in, beta=beta_in, gamma=gamma_in)
    st.markdown("---")
    show_bone = st.checkbox("Show bone model", value=True)
    show_xyproj = st.checkbox("Show XY floor projection", value=True)

with left:
    progress = st.slider("Progress (0 none → 3 full)", 0.0, 3.0, 0.0, 0.01)
    alpha = st.session_state["targets"]["alpha"]
    beta  = st.session_state["targets"]["beta"]
    gamma = st.session_state["targets"]["gamma"]
    R = staged_rotation(alpha, beta, gamma, progress)

    A_prox = np.array([0,1,0])
    A_dist = normalize(R @ A_prox)
    phi_xy = angle_xy(A_prox, A_dist)
    st.metric("Apparent torsion (XY) [deg]", f"{phi_xy:.2f}")

    fig = build_scene(R, show_bone=show_bone, show_xyproj=show_xyproj,
                      title=f"Stage progress={progress:.2f}")
    st.plotly_chart(fig, use_container_width=True)
