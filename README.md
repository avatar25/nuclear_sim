# Hybrid Nuclear Simulator

This project pairs a Rust simulation core with a Python dashboard:

- Rust uses Bevy's `App` and ECS scheduling layer to run a chain-reaction simulation.
- PyO3 exposes the simulator to Python as a native extension module.
- Streamlit renders the live dashboard, including neutron-population charts, core-density heat maps, and a control rod that drops into the reactor view.

## Project Layout

- `src/lib.rs` exposes the `NuclearSimulator` Python class.
- `src/simulation.rs` contains the Bevy-driven particle engine and multithreaded neutron update loop.
- `streamlit_app.py` is the real-time dashboard.
- `.streamlit/config.toml` tunes the neon-heavy theme.

## What the Simulator Does

- Models `Uranium235Atom` sites across a fissile lattice.
- Advances neutron particles in parallel across worker threads.
- Triggers fission when neutrons strike reactive fuel, emitting fission fragments and 2-3 new neutrons.
- Tracks live neutron population, reaction rate, and thermal diffusion.
- Animates a control rod that suppresses the reaction by absorbing neutrons and lowering local fission probability.

## Local Setup

The current workspace did not have `cargo`, `rustc`, or `maturin` installed when this scaffold was created, so the commands below were not executed here.

1. Install Rust with `rustup` so `cargo` and `rustc` are available.
2. Create and activate a Python environment.
3. Install the project with maturin:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
maturin develop
```

If you prefer `pip install -e .`, use `python -m pip install -e . --no-build-isolation` after installing `maturin` in the venv. That skips pip's temporary build environment and avoids re-downloading build tooling.

The editable install also pulls the dashboard dependencies from `pyproject.toml` (`numpy`, `pandas`, `plotly`, and `streamlit`). If your environment is offline, install those from local wheels first or use an environment where they are already available.

4. Launch the dashboard:

```bash
streamlit run /Users/shiben/Desktop/nuclear_sim/streamlit_app.py
```

Or use the one-shot launcher:

```bash
/Users/shiben/Desktop/nuclear_sim/run_sim.sh
```

## Notes

- The Rust core is capped at `150,000` neutrons by default to keep the dashboard responsive.
- `st.fragment(run_every=...)` keeps the Streamlit UI updating without blocking the rest of the app.
- The dashboard exposes slow-motion controls and a manual rod-depth target so you can watch suppression happen in real time.
