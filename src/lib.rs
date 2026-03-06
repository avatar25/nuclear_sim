mod simulation;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;
use simulation::{SimulationConfig, SimulationEngine, SimulationFrame};

#[pyclass(unsendable)]
struct NuclearSimulator {
    engine: SimulationEngine,
}

#[pymethods]
impl NuclearSimulator {
    #[new]
    #[pyo3(signature = (
        width = 1_000.0,
        height = 1_000.0,
        grid_width = 48,
        grid_height = 48,
        initial_neutrons = 320,
        fuel_density = 0.88,
        enrichment = 0.92,
        max_neutrons = 150_000
    ))]
    fn new(
        width: f32,
        height: f32,
        grid_width: usize,
        grid_height: usize,
        initial_neutrons: usize,
        fuel_density: f32,
        enrichment: f32,
        max_neutrons: usize,
    ) -> Self {
        let config = SimulationConfig::new(
            width,
            height,
            grid_width,
            grid_height,
            initial_neutrons,
            fuel_density,
            enrichment,
            max_neutrons,
        );

        Self {
            engine: SimulationEngine::new(config),
        }
    }

    fn start(&mut self) {
        self.engine.set_running(true);
    }

    fn stop(&mut self) {
        self.engine.set_running(false);
    }

    fn is_running(&self) -> bool {
        self.engine.is_running()
    }

    #[pyo3(signature = (steps = 4, dt = 0.0125, sample_limit = 4_096))]
    fn advance(&mut self, steps: usize, dt: f32, sample_limit: usize) -> SimulationFrame {
        self.engine.step_batch(steps, dt, sample_limit)
    }

    #[pyo3(signature = (steps = 4, dt = 0.0125, sample_limit = 4_096))]
    fn advance_if_running(&mut self, steps: usize, dt: f32, sample_limit: usize) -> SimulationFrame {
        self.engine.advance_if_running(steps, dt, sample_limit)
    }

    #[pyo3(signature = (sample_limit = 4_096))]
    fn snapshot(&self, sample_limit: usize) -> SimulationFrame {
        self.engine.snapshot(sample_limit)
    }

    fn drop_control_rod(&mut self) {
        self.engine.set_control_rod_target(1.0);
    }

    fn lift_control_rod(&mut self) {
        self.engine.set_control_rod_target(0.0);
    }

    fn set_control_rod_depth(&mut self, depth: f32) {
        self.engine.set_control_rod_target(depth);
    }

    fn set_control_rod_x(&mut self, x: f32) {
        self.engine.set_control_rod_x(x);
    }

    fn thread_count(&self) -> usize {
        self.engine.thread_count()
    }

    #[pyo3(signature = (initial_neutrons = None, fuel_density = None, enrichment = None))]
    fn reset(
        &mut self,
        initial_neutrons: Option<usize>,
        fuel_density: Option<f32>,
        enrichment: Option<f32>,
    ) -> SimulationFrame {
        self.engine
            .reset(initial_neutrons, fuel_density, enrichment);
        self.engine.snapshot(4_096)
    }
}

#[pymodule]
fn nuclear_sim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NuclearSimulator>()?;
    m.add_class::<SimulationFrame>()?;
    Ok(())
}
