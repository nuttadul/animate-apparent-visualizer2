
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Math helpers
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

# Geometry helpers
def cylinder_between(z0, z1, radius=0.3, n_theta=40, n_z=16):
    thetas = np.linspace(0, 2*np.pi, n_theta)
    zs = np.linspace(z0, z1, n_z)
    T, Z = np.meshgrid(thetas, zs)
    x = radius * np.cos(T); y = radius * np.sin(T); z = Z
    return x, y, z
def transform_points(R, pts):
    return (R @ pts.T).T
def anterior_stripe_lines(z0, z1, radius=0.3, theta=np.pi/2, n=50):
    zs = np.linspace(z0, z1, n)
    xs = radius*np.cos(theta)*np.ones_like(zs)
    ys = radius*np.sin(theta)*np.ones_like(zs)
    return np.stack([xs, ys, zs], axis=1)

def build_traces(R_dist, show_bone=True, show_stick=True, show_xyproj=True):
    traces = []
    # Proximal surface (fixed)
    xp, yp, zp = cylinder_between(0.0, 1.5, 0.3)
    traces.append(go.Surface(x=xp, y=yp, z=zp, showscale=False,
                             colorscale=[[0, "#d9e4ff"],[1, "#d9e4ff"]], visible=show_bone, name="prox"))
    # Prox stripe
    sp = anterior_stripe_lines(0.0, 1.5, 0.3)
    traces.append(go.Scatter3d(x=sp[:,0], y=sp[:,1], z=sp[:,2], mode="lines",
                               line=dict(color="crimson", width=8), visible=show_bone, name="prox_stripe"))
    # Distal surface (rotated)
    xd, yd, zd = cylinder_between(0.0, -1.5, 0.3)
    pts_d = np.stack([xd.flatten(), yd.flatten(), zd.flatten()], axis=1)
    td = transform_points(R_dist, pts_d)
    xd2 = td[:,0].reshape(xd.shape); yd2 = td[:,1].reshape(yd.shape); zd2 = td[:,2].reshape(zd.shape)
    traces.append(go.Surface(x=xd2, y=yd2, z=zd2, showscale=False,
                             colorscale=[[0, "#ffe0cc"],[1, "#ffe0cc"]], visible=show_bone, name="dist"))
    # Dist stripe
    sd = anterior_stripe_lines(0.0, -1.5, 0.3)
    tsd = transform_points(R_dist, sd)
    traces.append(go.Scatter3d(x=tsd[:,0], y=tsd[:,1], z=tsd[:,2], mode="lines",
                               line=dict(color="crimson", width=8), visible=show_bone, name="dist_stripe"))
    # Stick vectors
    origin = np.zeros(3); Zp = np.array([0,0,1]); Ap = np.array([0,1,0])
    Zd = normalize(R_dist @ np.array([0,0,-1])); Ad = normalize(R_dist @ np.array([0,1,0]))
    def arrow(start, vec, color, name):
        s = np.array(start); v = np.array(vec); tip = s + v
        return go.Scatter3d(x=[s[0], tip[0]], y=[s[1], tip[1]], z=[s[2], tip[2]],
                            mode="lines", line=dict(color=color, width=12), visible=show_stick, name=name)
    traces += [
        arrow(origin, Zp, "royalblue",  "Z_prox"),
        arrow(origin, Zd, "darkorange", "Z_dist"),
        arrow(origin, Ap*0.9, "seagreen", "A_prox"),
        arrow(origin, Ad*0.9, "crimson",  "A_dist"),
    ]
    # XY floor + projections
    limit=3.0
    rng=np.linspace(-limit, limit, 2); Xf,Yf=np.meshgrid(rng, rng); Zf=np.full_like(Xf,-limit)
    traces.append(go.Surface(x=Xf, y=Yf, z=Zf, opacity=0.12, showscale=False, visible=show_xyproj, name="floor"))
    zproj = -limit + 1e-3
    traces.append(go.Scatter3d(x=[0, Ap[0]*0.9], y=[0, Ap[1]*0.9], z=[zproj, zproj],
                               mode="lines", line=dict(color="rgba(46,139,87,0.65)", width=6), visible=show_xyproj, name="Aprox→XY"))
    traces.append(go.Scatter3d(x=[0, Ad[0]*0.9], y=[0, Ad[1]*0.9], z=[zproj, zproj],
                               mode="lines", line=dict(color="rgba(220,20,60,0.75)", width=6), visible=show_xyproj, name="Adist→XY"))
    return traces

def staged_rotation(alpha, beta, gamma, t):
    if t <= 0: return np.eye(3)
    if t < 1.0: return rot_z(gamma * t)
    elif t < 2.0: return rot_x(alpha * (t-1.0)) @ rot_z(gamma)
    else: return rot_y(beta * (t-2.0)) @ rot_x(alpha) @ rot_z(gamma)

# App
st.set_page_config(page_title="Apparent Torsion Visualizer – Animation", layout="wide")
st.title("Apparent Torsion Visualizer – Animation (Larger Model)")
st.caption("Play through torsion → sagittal → coronal, or scrub with the slider. Larger model view.")

alpha = st.slider("Sagittal (about X) [deg]", -60.0, 60.0, 0.0, 0.5)
beta  = st.slider("Coronal (about Y) [deg]", -60.0, 60.0, 0.0, 0.5)
gamma = st.slider("Torsion (about Z) [deg]", -90.0, 90.0, 0.0, 0.5)

show_bone  = st.checkbox("Show bone", value=True)
show_stick = st.checkbox("Show stick", value=True)
show_xyproj = st.checkbox("Show XY floor", value=True)

R_final = staged_rotation(alpha, beta, gamma, 3.0)
A_prox = np.array([0,1,0]); A_dist_final = normalize(R_final @ A_prox)
phi_final = angle_xy(A_prox, A_dist_final)
st.metric("Final apparent torsion (XY)", f"{phi_final:.2f}°")

limit=3.0
R0 = staged_rotation(alpha, beta, gamma, 0.0)
fig = go.Figure(data=build_traces(R0, show_bone, show_stick, show_xyproj))

n_frames = 60
frames = []
for i in range(n_frames+1):
    t = 3.0 * i / n_frames
    R = staged_rotation(alpha, beta, gamma, t)
    frames.append(go.Frame(data=build_traces(R, show_bone, show_stick, show_xyproj), name=f"{t:.3f}"))
fig.frames = frames

steps = []
for i in range(n_frames+1):
    t = 3.0 * i / n_frames
    steps.append(dict(method="animate",
                      args=[[f"{t:.3f}"], {"mode":"immediate",
                                           "frame":{"duration":0,"redraw":True},
                                           "transition":{"duration":0}}],
                      label=f"{t:.2f}"))
sliders = [dict(active=0, steps=steps, x=0.1, y=0.05, len=0.8,
                currentvalue=dict(prefix="Progress: ", suffix=" (0→3)", font=dict(size=14)))]

updatemenus = [dict(type="buttons", showactive=False, x=0.1, y=0.12,
                    buttons=[dict(label="Play", method="animate",
                                  args=[None, {"fromcurrent":True, "frame":{"duration":40,"redraw":True},
                                               "transition":{"duration":0}}]),
                             dict(label="Pause", method="animate",
                                  args=[[None], {"mode":"immediate",
                                                 "frame":{"duration":0,"redraw":False},
                                                 "transition":{"duration":0}}])])]

fig.update_layout(scene=dict(xaxis=dict(range=[-limit,limit]),
                             yaxis=dict(range=[-limit,limit]),
                             zaxis=dict(range=[-limit,limit]),
                             aspectmode="cube",
                             bgcolor="rgb(248,248,248)"),
                  paper_bgcolor="white",
                  margin=dict(l=0,r=0,t=30,b=0),
                  sliders=sliders, updatemenus=updatemenus)

st.plotly_chart(fig, use_container_width=True)
