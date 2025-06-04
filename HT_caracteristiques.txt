import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation
import tempfile
from PIL import Image



def simuler(Hres, K, L, D, a, f, n, Z0, Z1, Q0, tf, T):
    A = (math.pi * D**2) / 4 
    R = f / (2 * D * A)
    dx = L / float(n) 
    dt = dx / a     
    ca = 9.81 * A / a     
    m = n + 1 
    p = round(T / dt) + 2 
    
    thau = np.zeros((p), dtype=float)
    Q = np.zeros((m, p), dtype=float)
    H = np.zeros((m, p), dtype=float)
    h = np.zeros((m, p), dtype=float)
    cn = np.zeros((m, p+1), dtype=float)
    cp = np.zeros((m, p+1), dtype=float)
    cv = np.zeros((p), dtype=float)
    
    for i in range(m):
        Q[i][0] = Q0
        H[i][0] = Hres - (K + 1 + f * dx * i / D) * ((Q0 / A)**2) / (2 * 9.81)
        h[i][0] = H[i][0] - Z0 - ((Z0 - Z1) * dx * i) / L                            
    
    for i in range(m):
        if i != n:
            Q[i][1] = Q0
            H[i][1] = H[i][0]
            h[i][1] = h[i][0]
        else:
            Q[n][1] = 0 if tf == 0 else Q0    
            H[n][1] = H[n][0]
            h[i][1] = h[i][0]
    
    for i in range(n):
        cn[i][2] = Q[i+1][1] - ca * H[i+1][1] - R * Q[i+1][1] * abs(Q[i+1][1]) * dt
        cp[i+1][2] = Q[i][1] + ca * H[i][1] - R * Q[i][1] * abs(Q[i][1]) * dt
    
    for j in range(2, round(T/dt)+2): 
        if tf == 0:
            thau[j-2] = 0
        elif dt * (j-2) < tf:
            thau[j-2] = 1 - (j-1) * dt / tf
        else:
            thau[j-2] = 0          
        
        for i in range(m):
            if i == 0: 
                k1 = ca * (1 + K) / (2 * 9.81 * (A**2))
                Q[i][j] = (-1 + (1 + 4 * k1 * (cn[i][j] + ca * Hres))**0.5) / (2 * k1)
                if Q[0][j] >= 0:
                    H[i][j] = Hres - (1 + K) * ((Q[i][j])**2) / (2 * 9.81 * A**2)
                    h[i][j] = H[i][j] - Z0 - ((Z0 - Z1) * dx * i) / L
                else:
                    k1 = ca * (1 - K) / (2 * 9.81 * (A**2))
                    Q[i][j] = (-1 + (1 + 4 * k1 * (cn[i][j] + ca * Hres))**0.5) / (2 * k1)
                    H[i][j] = Hres - (1 - K) * ((Q[i][j])**2) / (2 * 9.81 * A**2)
                    h[i][j] = H[i][j] - Z0 - ((Z0 - Z1) * dx * i) / L
            elif i == n: 
                cv[j-2] = (thau[j-2] * Q[n][1])**2 / (ca * H[n][1])
                Q[i][j] = 0.5 * (-cv[j-2] + (cv[j-2]**2 + 4 * cp[i][j] * cv[j-2])**0.5)
                H[i][j] = (cp[i][j] - Q[i][j]) / ca
                h[i][j] = H[i][j] - Z0 - ((Z0 - Z1) * dx * i) / L
            else: 
                Q[i][j] = (cp[i][j] + cn[i][j]) / 2
                H[i][j] = (Q[i][j] - cn[i][j]) / (ca)
                h[i][j] = H[i][j] - Z0 - ((Z0 - Z1) * dx * i) / L
        
        for i in range(1, m):
            cp[i][j+1] = Q[i-1][j] + ca * H[i-1][j] - R * Q[i-1][j] * abs(Q[i-1][j]) * dt
        for i in range(n):
            cn[i][j+1] = Q[i+1][j] - ca * H[i+1][j] - R * Q[i+1][j] * abs(Q[i+1][j]) * dt
    
    t = np.array([i * dt for i in range(p)])
    x1 = np.array([i * dx for i in range(m)])
    return t, x1, H, h, Q, dt, dx


st.set_page_config(layout="wide", page_title="Simulation Transitoire Hydraulique")
st.title("💧 Simulation de Transitoire Hydraulique Réservoir-Conduite-Vanne")

col1, col2 = st.columns(2)

with col1:
    st.header("🔧 Paramètres de Simulation")
    Hres = st.number_input("Charge du réservoir (Hres, m)", value=80.0)
    K = st.number_input("Coefficient de perte de charge (K)", value=0.5)
    L = st.number_input("Longueur de la conduite (L, m)", value=3750.0)
    D = st.number_input("Diamètre de la conduite (D, m)", value=0.4)
    a = st.number_input("Célérité de l'onde (a, m/s)", value=1250.0)
    f = st.number_input("Coefficient de frottement (f)", value=0.05)
    n = st.number_input("Nombre de segments (n)", value=3, min_value=2, max_value=200)

with col2:
    st.header("⛓️ Conditions aux Limites")
    Z0 = st.number_input("Altitude amont (Z0, m)", value=0.0)
    Z1 = st.number_input("Altitude aval (Z1, m)", value=0.0)
    Q0 = st.number_input("Débit initial (Q0, m³/s)", value=0.175)
    tf = st.number_input("Temps de fermeture de la vanne (tf, s)", value=1.0)
    T = st.number_input("Durée de simulation (T, s)", value=10.0)

    positions = [round(i * (L / int(n)), 2) for i in range(int(n) + 1)]
    x_pos = st.selectbox("Position d’analyse (x, m)", options=positions, index=len(positions) // 2)


if st.button("🚀 Lancer la Simulation"):
    with st.spinner("Simulation en cours..."):
        t, x1, H, h, Q, dt, dx = simuler(Hres, K, L, D, a, f, int(n), Z0, Z1, Q0, tf, T)
        x_index = np.argmin(np.abs(x1 - x_pos))
        st.success(f"✅ Simulation terminée ! Analyse à x = {x1[x_index]:.1f} m")

        def create_gif(data, ylabel, title, color):
            fig, ax = plt.subplots()
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(min(data[x_index]), max(data[x_index]))
            ax.set_title(title)
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel(ylabel)
            line, = ax.plot([], [], color=color)

            def animate(i):
                line.set_data(t[:i], data[x_index][:i])
                return line,

            ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            ani.save(temp_file.name, writer='pillow')
            plt.close(fig)
            return temp_file.name

        st.header("📽️ Animation des Résultats")

        gif1 = create_gif(H, "H (m)", "Évolution de la Charge H(t)", 'blue')
        gif2 = create_gif(h, "h (m)", "Évolution de la Pression h(t)", 'green')
        gif3 = create_gif(Q, "Q (m³/s)", "Évolution du Débit Q(t)", 'red')

        st.image(gif1, caption="Charge H(t)", use_column_width=True)
        st.image(gif2, caption="Pression h(t)", use_column_width=True)
        st.image(gif3, caption="Débit Q(t)", use_column_width=True)

        st.header("📌 Résumé Numérique")
        colg1, colg2 = st.columns(2)
        with colg1:
            st.metric("H max (m)", f"{max(H[x_index]):.2f}")
            st.metric("h max (m)", f"{max(h[x_index]):.2f}")
        with colg2:
            st.metric("Q max (m³/s)", f"{max(Q[x_index]):.4f}")
            st.metric("Q min (m³/s)", f"{min(Q[x_index]):.4f}")

