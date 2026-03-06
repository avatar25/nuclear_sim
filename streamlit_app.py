from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nuclear_sim_rs import NuclearSimulator, SimulationFrame


st.set_page_config(
    page_title="Hybrid Nuclear Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEON_CYAN = "#00F5D4"
NEON_PINK = "#FF4D8D"
NEON_ORANGE = "#FF9F1C"
SURFACE_BG = "#060816"
PANEL_BG = "rgba(16, 23, 42, 0.82)"


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --glow-cyan: {NEON_CYAN};
                --glow-pink: {NEON_PINK};
                --glow-orange: {NEON_ORANGE};
                --panel: {PANEL_BG};
                --ink: {SURFACE_BG};
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top, rgba(0, 245, 212, 0.16), transparent 32%),
                    radial-gradient(circle at 85% 15%, rgba(255, 77, 141, 0.18), transparent 24%),
                    linear-gradient(180deg, #0a1022 0%, #060816 42%, #02040a 100%);
                color: #f8fafc;
                font-family: "Space Grotesk", "Avenir Next", sans-serif;
            }}

            section[data-testid="stSidebar"] {{
                background: rgba(10, 16, 34, 0.92);
                border-right: 1px solid rgba(0, 245, 212, 0.14);
            }}

            div[data-testid="stMetric"] {{
                background: rgba(10, 16, 34, 0.66);
                border: 1px solid rgba(0, 245, 212, 0.14);
                border-radius: 18px;
                padding: 0.75rem 1rem;
                box-shadow: 0 0 24px rgba(0, 245, 212, 0.06);
            }}

            div[data-testid="stMetricValue"] {{
                color: var(--glow-cyan);
            }}

            div.stButton > button {{
                border-radius: 999px;
                border: 1px solid rgba(0, 245, 212, 0.24);
                background: linear-gradient(135deg, rgba(0, 245, 212, 0.16), rgba(255, 77, 141, 0.12));
                color: #f8fafc;
                transition: transform 180ms ease, box-shadow 180ms ease;
            }}

            div.stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 0 18px rgba(0, 245, 212, 0.16);
            }}

            .reactor-card {{
                background: rgba(10, 16, 34, 0.72);
                border: 1px solid rgba(0, 245, 212, 0.14);
                border-radius: 24px;
                padding: 1rem 1.15rem;
                box-shadow: 0 24px 48px rgba(0, 0, 0, 0.22);
            }}

            .reactor-kicker {{
                letter-spacing: 0.24em;
                text-transform: uppercase;
                color: rgba(148, 163, 184, 0.9);
                font-size: 0.78rem;
                margin-bottom: 0.2rem;
            }}

            .reactor-title {{
                font-size: 3rem;
                line-height: 0.95;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }}

            .reactor-subtitle {{
                max-width: 50rem;
                color: rgba(226, 232, 240, 0.86);
                margin-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def bootstrap() -> NuclearSimulator:
    if "sim" not in st.session_state:
        st.session_state.sim = NuclearSimulator()
        st.session_state.frame = st.session_state.sim.snapshot(sample_limit=3_600)
        st.session_state.running = False
        st.session_state.slow_motion = False
        st.session_state.steps_per_refresh = 6
        st.session_state.dt = 0.012
        st.session_state.sample_limit = 3_600
        st.session_state.rod_target = float(st.session_state.frame.rod_depth)
        st.session_state.rod_x = float(st.session_state.frame.rod_x)
        st.session_state.initial_neutrons = 320
        st.session_state.fuel_density = 0.88
        st.session_state.enrichment = 0.92
    return st.session_state.sim


def history_frame(frame: SimulationFrame) -> pd.DataFrame:
    if not frame.history_ticks:
        return pd.DataFrame({"tick": [frame.tick], "neutron_population": [frame.neutron_population]})

    return pd.DataFrame(
        {
            "tick": frame.history_ticks,
            "neutron_population": frame.history_population,
            "average_heat": frame.history_average_heat,
            "reaction_rate": frame.history_reaction_rate,
        }
    )


def density_array(frame: SimulationFrame) -> np.ndarray:
    return np.array(frame.density_map, dtype=float).reshape(frame.grid_height, frame.grid_width)


def heat_array(frame: SimulationFrame) -> np.ndarray:
    return np.array(frame.heat_map, dtype=float).reshape(frame.grid_height, frame.grid_width)


def particle_colors(frame: SimulationFrame) -> list[str]:
    colors: list[str] = []
    for kind, energy in zip(frame.sample_kind, frame.sample_energy):
        if kind == 0:
            alpha = min(0.95, 0.45 + energy * 0.18)
            colors.append(f"rgba(0,245,212,{alpha:.3f})")
        else:
            alpha = min(0.92, 0.32 + energy * 0.22)
            colors.append(f"rgba(255,159,28,{alpha:.3f})")
    return colors


def particle_sizes(frame: SimulationFrame) -> list[float]:
    sizes: list[float] = []
    for kind, energy in zip(frame.sample_kind, frame.sample_energy):
        base = 6.5 if kind == 0 else 9.0
        sizes.append(base + energy * 2.6)
    return sizes


def build_reactor_figure(frame: SimulationFrame) -> go.Figure:
    density = density_array(frame)
    x_coords = np.linspace(0.0, frame.core_width, frame.grid_width)
    y_coords = np.linspace(0.0, frame.core_height, frame.grid_height)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_coords,
            y=y_coords,
            z=density,
            colorscale=[
                [0.0, "#030712"],
                [0.18, "#11213E"],
                [0.38, "#164E63"],
                [0.58, "#00A7B6"],
                [0.78, "#FFD166"],
                [1.0, "#FF4D8D"],
            ],
            opacity=0.86,
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=frame.sample_x,
            y=frame.sample_y,
            mode="markers",
            marker=dict(
                size=particle_sizes(frame),
                color=particle_colors(frame),
                line=dict(width=1.0, color="rgba(255,255,255,0.16)"),
            ),
            hoverinfo="skip",
        )
    )

    rod_top = frame.core_height
    rod_bottom = frame.core_height * (1.0 - frame.rod_depth)
    fig.add_shape(
        type="rect",
        x0=frame.rod_x - frame.rod_width * 0.5,
        x1=frame.rod_x + frame.rod_width * 0.5,
        y0=rod_bottom,
        y1=rod_top,
        fillcolor="rgba(226, 232, 240, 0.18)",
        line=dict(color="rgba(226, 232, 240, 0.85)", width=2),
    )

    fig.add_annotation(
        x=frame.rod_x,
        y=frame.core_height + frame.core_height * 0.045,
        text="CONTROL ROD",
        showarrow=False,
        font=dict(color="#E2E8F0", size=12),
    )

    fig.update_layout(
        title=dict(text="Core Density + Neon Particle Field", font=dict(size=20, color="#F8FAFC")),
        margin=dict(l=18, r=18, t=54, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5, 8, 22, 0.94)",
        xaxis=dict(
            range=[0, frame.core_width],
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            range=[0, frame.core_height + frame.core_height * 0.08],
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        height=620,
    )
    return fig


def build_heat_figure(frame: SimulationFrame) -> go.Figure:
    heat = heat_array(frame)
    x_coords = np.linspace(0.0, frame.core_width, frame.grid_width)
    y_coords = np.linspace(0.0, frame.core_height, frame.grid_height)
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                x=x_coords,
                y=y_coords,
                z=heat,
                colorscale="Turbo",
                opacity=0.95,
                showscale=False,
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Thermal Bloom", font=dict(size=18, color="#F8FAFC")),
        margin=dict(l=16, r=16, t=44, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5, 8, 22, 0.94)",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        height=280,
    )
    return fig


def control_panel(sim: NuclearSimulator, frame: SimulationFrame) -> None:
    with st.sidebar:
        st.subheader("Reactor Controls")
        st.caption(f"Rust worker threads: {sim.thread_count()} | particle ceiling: 150,000")

        start_col, stop_col = st.columns(2)
        if start_col.button("Start", width="stretch", disabled=st.session_state.running):
            st.session_state.running = True
            sim.start()
        if stop_col.button("Pause", width="stretch", disabled=not st.session_state.running):
            st.session_state.running = False
            sim.stop()

        st.toggle("Slow motion", key="slow_motion")
        st.slider("Steps per refresh", 1, 12, key="steps_per_refresh")
        st.slider("Time step", 0.004, 0.03, step=0.001, key="dt")
        st.slider("Particle sample budget", 800, 6_000, step=200, key="sample_limit")

        st.divider()
        st.subheader("Control Rod")
        st.slider(
            "Rod lane",
            0.0,
            float(frame.core_width),
            key="rod_x",
            help="Slide the insertion lane before dropping the rod.",
        )
        sim.set_control_rod_x(float(st.session_state.rod_x))

        rod_col_a, rod_col_b = st.columns(2)
        if rod_col_a.button("Drop Rod", width="stretch", type="primary"):
            st.session_state.rod_target = 1.0
            sim.drop_control_rod()
        if rod_col_b.button("Lift Rod", width="stretch"):
            st.session_state.rod_target = 0.0
            sim.lift_control_rod()

        st.slider("Target depth", 0.0, 1.0, step=0.02, key="rod_target")
        sim.set_control_rod_depth(float(st.session_state.rod_target))

        st.divider()
        st.subheader("Fuel Lattice")
        with st.form("reset-core"):
            st.slider("Seed neutrons", 64, 2_400, step=32, key="initial_neutrons")
            st.slider("Fuel density", 0.2, 1.0, step=0.02, key="fuel_density")
            st.slider("U-235 enrichment", 0.2, 1.0, step=0.02, key="enrichment")
            submitted = st.form_submit_button("Re-seed Core", width="stretch")

        if submitted:
            st.session_state.running = False
            sim.stop()
            st.session_state.frame = sim.reset(
                initial_neutrons=st.session_state.initial_neutrons,
                fuel_density=st.session_state.fuel_density,
                enrichment=st.session_state.enrichment,
            )
            st.session_state.rod_target = float(st.session_state.frame.rod_depth)
            st.session_state.rod_x = float(st.session_state.frame.rod_x)


def render_metrics(frame: SimulationFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Neutron Population", f"{frame.neutron_population:,}")
    col2.metric("Average Heat", f"{frame.average_heat:0.2f}")
    col3.metric("Reaction Rate", f"{frame.reaction_rate:0.1f}/s")
    col4.metric("Rod Insertion", f"{frame.rod_depth * 100:0.0f}%")


inject_theme()
sim = bootstrap()

st.markdown(
    """
    <div class="reactor-card">
        <div class="reactor-kicker">Bevy + PyO3 + Streamlit</div>
        <div class="reactor-title">Hybrid Nuclear Simulator</div>
        <div class="reactor-subtitle">
            Neon particle traces ride over a live core-density heat map while a control rod can be
            dropped into the reactor lane to choke off the chain reaction in real time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

control_panel(sim, st.session_state.frame)

run_every = 0.18 if st.session_state.running else None


@st.fragment(run_every=run_every)
def reactor_fragment() -> None:
    sim_ref = st.session_state.sim
    if st.session_state.running:
        live_dt = 0.006 if st.session_state.slow_motion else float(st.session_state.dt)
        live_steps = 1 if st.session_state.slow_motion else int(st.session_state.steps_per_refresh)
        frame = sim_ref.advance_if_running(
            steps=live_steps,
            dt=live_dt,
            sample_limit=int(st.session_state.sample_limit),
        )
    else:
        frame = sim_ref.snapshot(sample_limit=int(st.session_state.sample_limit))

    st.session_state.frame = frame

    render_metrics(frame)
    plot_col, data_col = st.columns([1.7, 1.0], gap="large")
    with plot_col:
        st.plotly_chart(
            build_reactor_figure(frame),
            width="stretch",
            config={"displayModeBar": False},
        )
    with data_col:
        st.markdown("#### Neutron Population")
        st.line_chart(
            history_frame(frame).set_index("tick")[["neutron_population"]],
            height=250,
            width="stretch",
        )
        st.plotly_chart(
            build_heat_figure(frame),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "Drop the control rod to suppress reactivity. Slow motion lowers the frame delta so the "
            "suppression wave is easier to watch as the neutron count collapses."
        )


reactor_fragment()
