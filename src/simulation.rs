use std::f32::consts::TAU;
use std::mem;
use std::thread;

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use bevy_math::Vec2;
use pyo3::prelude::*;

const HISTORY_CAP: usize = 240;
const FRAGMENT_LIFETIME: f32 = 0.8;

#[derive(Clone, Copy, Debug, Resource)]
pub struct SimulationConfig {
    pub width: f32,
    pub height: f32,
    pub grid_width: usize,
    pub grid_height: usize,
    pub initial_neutrons: usize,
    pub fuel_density: f32,
    pub enrichment: f32,
    pub max_neutrons: usize,
    pub neutron_speed: f32,
    pub neutron_lifetime: f32,
    pub base_fission_probability: f32,
    pub cooling_factor: f32,
    pub rod_width: f32,
    pub rod_drop_speed: f32,
    pub worker_count: usize,
}

impl SimulationConfig {
    pub fn new(
        width: f32,
        height: f32,
        grid_width: usize,
        grid_height: usize,
        initial_neutrons: usize,
        fuel_density: f32,
        enrichment: f32,
        max_neutrons: usize,
    ) -> Self {
        let worker_count = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4)
            .max(2);

        Self {
            width,
            height,
            grid_width: grid_width.max(16),
            grid_height: grid_height.max(16),
            initial_neutrons,
            fuel_density: fuel_density.clamp(0.1, 1.0),
            enrichment: enrichment.clamp(0.1, 1.0),
            max_neutrons: max_neutrons.max(initial_neutrons.max(1)),
            neutron_speed: (width.max(height) * 0.06).max(6.0),
            neutron_lifetime: 4.5,
            base_fission_probability: 0.42,
            cooling_factor: 0.93,
            rod_width: width * 0.09,
            rod_drop_speed: 0.95,
            worker_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Resource)]
struct FrameTiming {
    dt: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Uranium235Atom {
    pub enrichment: f32,
    pub fissile_mass: f32,
    pub stability: f32,
    pub active: bool,
}

impl Uranium235Atom {
    fn seeded(config: SimulationConfig, x: usize, y: usize) -> Self {
        let cx = (config.grid_width as f32 - 1.0) * 0.5;
        let cy = (config.grid_height as f32 - 1.0) * 0.5;
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let radial = (dx * dx + dy * dy).sqrt() / cx.max(cy).max(1.0);
        let radial_falloff = (1.08 - radial).clamp(0.0, 1.0);
        let noise = random01(((x as u64) << 32) ^ y as u64 ^ 0xA5A5_5A5A);
        let occupancy = (config.fuel_density * radial_falloff * (0.75 + 0.35 * noise)).clamp(0.0, 1.0);

        Self {
            enrichment: (config.enrichment * (0.9 + 0.15 * noise)).clamp(0.1, 1.0),
            fissile_mass: occupancy,
            stability: (0.82 - radial * 0.1).clamp(0.45, 0.92),
            active: occupancy > 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NeutronParticle {
    id: u64,
    position: Vec2,
    velocity: Vec2,
    energy: f32,
    age: f32,
}

#[derive(Clone, Copy, Debug)]
struct FissionFragment {
    position: Vec2,
    velocity: Vec2,
    energy: f32,
    life: f32,
}

#[derive(Clone, Copy, Debug, Resource)]
struct ControlRodState {
    x: f32,
    width: f32,
    depth: f32,
    target_depth: f32,
    drop_speed: f32,
}

impl ControlRodState {
    fn inserted_depth(self, core_height: f32) -> f32 {
        self.depth.clamp(0.0, 1.0) * core_height
    }

    fn absorbs(self, position: Vec2, core_height: f32) -> bool {
        let top_band = core_height - self.inserted_depth(core_height);
        (position.x - self.x).abs() <= self.width * 0.5 && position.y >= top_band
    }

    fn suppression_at(self, position: Vec2, core_height: f32) -> f32 {
        if self.depth <= 0.0 {
            return 0.0;
        }

        let horizontal = 1.0 - ((position.x - self.x).abs() / self.width.max(1.0)).clamp(0.0, 1.0);
        let vertical = if position.y >= core_height * (1.0 - self.depth) {
            1.0
        } else {
            (self.depth * 0.45).clamp(0.0, 0.45)
        };

        (horizontal * vertical).clamp(0.0, 0.96)
    }
}

#[derive(Resource)]
struct SimulationState {
    tick: u64,
    running: bool,
    next_particle_id: u64,
    rod: ControlRodState,
    fuel: Vec<Uranium235Atom>,
    neutrons: Vec<NeutronParticle>,
    fragments: Vec<FissionFragment>,
    heat: Vec<f32>,
    density: Vec<f32>,
    history_ticks: Vec<u64>,
    history_population: Vec<usize>,
    history_average_heat: Vec<f32>,
    history_reaction_rate: Vec<f32>,
    last_fissions: usize,
}

impl SimulationState {
    fn seeded(config: SimulationConfig) -> Self {
        let fuel_cells = config.grid_width * config.grid_height;
        let mut state = Self {
            tick: 0,
            running: false,
            next_particle_id: 1,
            rod: ControlRodState {
                x: config.width * 0.5,
                width: config.rod_width,
                depth: 0.0,
                target_depth: 0.0,
                drop_speed: config.rod_drop_speed,
            },
            fuel: Vec::with_capacity(fuel_cells),
            neutrons: Vec::with_capacity(config.initial_neutrons.saturating_mul(2)),
            fragments: Vec::new(),
            heat: vec![0.0; fuel_cells],
            density: vec![0.0; fuel_cells],
            history_ticks: Vec::with_capacity(HISTORY_CAP),
            history_population: Vec::with_capacity(HISTORY_CAP),
            history_average_heat: Vec::with_capacity(HISTORY_CAP),
            history_reaction_rate: Vec::with_capacity(HISTORY_CAP),
            last_fissions: 0,
        };

        for y in 0..config.grid_height {
            for x in 0..config.grid_width {
                state.fuel.push(Uranium235Atom::seeded(config, x, y));
            }
        }

        for seed_index in 0..config.initial_neutrons {
            let angle = random01(seed_index as u64 * 13 + 3) * TAU;
            let radius = random01(seed_index as u64 * 97 + 5) * config.width.min(config.height) * 0.08;
            let origin = Vec2::new(config.width * 0.5, config.height * 0.5)
                + Vec2::from_angle(angle) * radius;
            let velocity = Vec2::from_angle(angle + TAU * 0.25) * config.neutron_speed;
            let neutron = state.spawn_neutron(origin, velocity, 1.0);
            state.neutrons.push(neutron);
        }

        rebuild_density_buffers(&mut state, config);
        state.record_history(config, 0.0);
        state
    }

    fn spawn_neutron(&mut self, position: Vec2, velocity: Vec2, energy: f32) -> NeutronParticle {
        let neutron = NeutronParticle {
            id: self.next_particle_id,
            position,
            velocity,
            energy,
            age: 0.0,
        };
        self.next_particle_id += 1;
        neutron
    }

    fn record_history(&mut self, config: SimulationConfig, reaction_rate: f32) {
        self.history_ticks.push(self.tick);
        self.history_population.push(self.neutrons.len());
        self.history_average_heat.push(average_heat(&self.heat));
        self.history_reaction_rate.push(reaction_rate);

        while self.history_ticks.len() > HISTORY_CAP {
            self.history_ticks.remove(0);
            self.history_population.remove(0);
            self.history_average_heat.remove(0);
            self.history_reaction_rate.remove(0);
        }

        if self.neutrons.len() > config.max_neutrons {
            self.neutrons.truncate(config.max_neutrons);
        }
    }
}

#[pyclass]
pub struct SimulationFrame {
    #[pyo3(get)]
    pub tick: u64,
    #[pyo3(get)]
    pub neutron_population: usize,
    #[pyo3(get)]
    pub average_heat: f32,
    #[pyo3(get)]
    pub reaction_rate: f32,
    #[pyo3(get)]
    pub rod_depth: f32,
    #[pyo3(get)]
    pub rod_x: f32,
    #[pyo3(get)]
    pub rod_width: f32,
    #[pyo3(get)]
    pub core_width: f32,
    #[pyo3(get)]
    pub core_height: f32,
    #[pyo3(get)]
    pub grid_width: usize,
    #[pyo3(get)]
    pub grid_height: usize,
    #[pyo3(get)]
    pub sample_x: Vec<f32>,
    #[pyo3(get)]
    pub sample_y: Vec<f32>,
    #[pyo3(get)]
    pub sample_kind: Vec<u8>,
    #[pyo3(get)]
    pub sample_energy: Vec<f32>,
    #[pyo3(get)]
    pub density_map: Vec<f32>,
    #[pyo3(get)]
    pub heat_map: Vec<f32>,
    #[pyo3(get)]
    pub history_ticks: Vec<u64>,
    #[pyo3(get)]
    pub history_population: Vec<usize>,
    #[pyo3(get)]
    pub history_average_heat: Vec<f32>,
    #[pyo3(get)]
    pub history_reaction_rate: Vec<f32>,
}

pub struct SimulationEngine {
    app: App,
    worker_count: usize,
}

impl SimulationEngine {
    pub fn new(config: SimulationConfig) -> Self {
        let worker_count = config.worker_count;
        let mut app = App::new();
        app.insert_resource(config);
        app.insert_resource(FrameTiming { dt: 0.0125 });
        app.insert_resource(SimulationState::seeded(config));
        app.add_systems(
            Update,
            (
                animate_control_rod,
                advance_fragments,
                simulate_particles,
                diffuse_heat,
                rebuild_density,
                capture_metrics,
            )
                .chain(),
        );

        Self { app, worker_count }
    }

    pub fn set_running(&mut self, running: bool) {
        self.app.world_mut().resource_mut::<SimulationState>().running = running;
    }

    pub fn is_running(&self) -> bool {
        self.app.world().resource::<SimulationState>().running
    }

    pub fn set_control_rod_target(&mut self, depth: f32) {
        self.app.world_mut().resource_mut::<SimulationState>().rod.target_depth =
            depth.clamp(0.0, 1.0);
    }

    pub fn set_control_rod_x(&mut self, x: f32) {
        let world = self.app.world_mut();
        let max_x = {
            let config = world.resource::<SimulationConfig>();
            config.width
        };
        world.resource_mut::<SimulationState>().rod.x = x.clamp(0.0, max_x);
    }

    pub fn step_batch(&mut self, steps: usize, dt: f32, sample_limit: usize) -> SimulationFrame {
        self.app.world_mut().resource_mut::<FrameTiming>().dt = dt.clamp(0.001, 0.05);

        for _ in 0..steps.max(1) {
            self.app.update();
        }

        self.snapshot(sample_limit)
    }

    pub fn advance_if_running(
        &mut self,
        steps: usize,
        dt: f32,
        sample_limit: usize,
    ) -> SimulationFrame {
        if self.is_running() {
            self.step_batch(steps, dt, sample_limit)
        } else {
            self.snapshot(sample_limit)
        }
    }

    pub fn snapshot(&self, sample_limit: usize) -> SimulationFrame {
        let world = self.app.world();
        let config = *world.resource::<SimulationConfig>();
        let state = world.resource::<SimulationState>();
        build_frame(&state, config, sample_limit)
    }

    pub fn reset(
        &mut self,
        initial_neutrons: Option<usize>,
        fuel_density: Option<f32>,
        enrichment: Option<f32>,
    ) {
        let world = self.app.world_mut();
        {
            let mut config = world.resource_mut::<SimulationConfig>();
            if let Some(initial_neutrons) = initial_neutrons {
                config.initial_neutrons = initial_neutrons.max(1);
                config.max_neutrons = config.max_neutrons.max(config.initial_neutrons);
            }
            if let Some(fuel_density) = fuel_density {
                config.fuel_density = fuel_density.clamp(0.1, 1.0);
            }
            if let Some(enrichment) = enrichment {
                config.enrichment = enrichment.clamp(0.1, 1.0);
            }
        }

        let config = *world.resource::<SimulationConfig>();
        *world.resource_mut::<SimulationState>() = SimulationState::seeded(config);
    }

    pub fn thread_count(&self) -> usize {
        self.worker_count
    }
}

fn animate_control_rod(timing: Res<FrameTiming>, mut state: ResMut<SimulationState>) {
    let delta = timing.dt * state.rod.drop_speed;
    let difference = state.rod.target_depth - state.rod.depth;

    if difference.abs() <= delta {
        state.rod.depth = state.rod.target_depth;
    } else {
        state.rod.depth += difference.signum() * delta;
    }
}

fn advance_fragments(timing: Res<FrameTiming>, mut state: ResMut<SimulationState>) {
    let dt = timing.dt;
    state.fragments.retain_mut(|fragment| {
        fragment.position += fragment.velocity * dt;
        fragment.velocity *= 0.985;
        fragment.life -= dt;
        fragment.energy *= 0.992;
        fragment.life > 0.0
    });
}

fn simulate_particles(
    config: Res<SimulationConfig>,
    timing: Res<FrameTiming>,
    mut state: ResMut<SimulationState>,
) {
    state.tick = state.tick.saturating_add(1);
    state.last_fissions = 0;

    let dt = timing.dt;
    let config_value = *config;
    let fuel_snapshot = state.fuel.clone();
    let rod = state.rod;
    let neutrons = mem::take(&mut state.neutrons);

    let outcomes = process_parallel_chunks(&neutrons, &fuel_snapshot, config_value, rod, state.tick, dt);
    let mut next_neutrons = Vec::with_capacity(neutrons.len());
    let mut events = Vec::new();

    for outcome in outcomes {
        next_neutrons.extend(outcome.survivors);
        events.extend(outcome.fissions);
    }

    state.neutrons = next_neutrons;
    apply_fission_events(&mut state, config_value, events);
}

fn diffuse_heat(config: Res<SimulationConfig>, mut state: ResMut<SimulationState>) {
    let mut next = vec![0.0; state.heat.len()];

    for y in 0..config.grid_height {
        for x in 0..config.grid_width {
            let idx = y * config.grid_width + x;
            let mut sum = state.heat[idx] * 0.58;
            let mut weight = 0.58;

            for ny in y.saturating_sub(1)..=(y + 1).min(config.grid_height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(config.grid_width - 1) {
                    if nx == x && ny == y {
                        continue;
                    }

                    let neighbor = ny * config.grid_width + nx;
                    sum += state.heat[neighbor] * 0.052;
                    weight += 0.052;
                }
            }

            next[idx] = (sum / weight) * config.cooling_factor;
        }
    }

    state.heat = next;
}

fn rebuild_density(config: Res<SimulationConfig>, mut state: ResMut<SimulationState>) {
    rebuild_density_buffers(&mut state, *config);
}

fn capture_metrics(config: Res<SimulationConfig>, timing: Res<FrameTiming>, mut state: ResMut<SimulationState>) {
    let reaction_rate = state.last_fissions as f32 / timing.dt.max(0.001);
    state.record_history(*config, reaction_rate);
}

fn rebuild_density_buffers(state: &mut SimulationState, config: SimulationConfig) {
    for (idx, density) in state.density.iter_mut().enumerate() {
        let atom = state.fuel[idx];
        let base = if atom.active {
            atom.fissile_mass * 0.8 + atom.enrichment * 0.2
        } else {
            atom.fissile_mass * 0.1
        };
        *density = base + state.heat[idx] * 0.08;
    }

    for neutron in &state.neutrons {
        if let Some(idx) = cell_index(neutron.position, config) {
            state.density[idx] += 0.42;
        }
    }

    for fragment in &state.fragments {
        if let Some(idx) = cell_index(fragment.position, config) {
            state.density[idx] += 0.18;
        }
    }
}

#[derive(Default)]
struct ChunkOutcome {
    survivors: Vec<NeutronParticle>,
    fissions: Vec<FissionEvent>,
}

#[derive(Clone, Copy)]
struct FissionEvent {
    cell_idx: usize,
    position: Vec2,
    velocity: Vec2,
    source_id: u64,
    energy: f32,
}

fn process_parallel_chunks(
    neutrons: &[NeutronParticle],
    fuel: &[Uranium235Atom],
    config: SimulationConfig,
    rod: ControlRodState,
    tick: u64,
    dt: f32,
) -> Vec<ChunkOutcome> {
    if neutrons.is_empty() {
        return Vec::new();
    }

    let worker_count = config.worker_count.max(1).min(neutrons.len());
    let chunk_size = (neutrons.len() + worker_count - 1) / worker_count;
    let mut outcomes = Vec::with_capacity(worker_count);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in neutrons.chunks(chunk_size).enumerate() {
            handles.push(scope.spawn(move || {
                process_chunk(chunk, fuel, config, rod, tick, dt, chunk_index as u64)
            }));
        }

        for handle in handles {
            outcomes.push(handle.join().expect("nuclear worker panicked"));
        }
    });

    outcomes
}

fn process_chunk(
    neutrons: &[NeutronParticle],
    fuel: &[Uranium235Atom],
    config: SimulationConfig,
    rod: ControlRodState,
    tick: u64,
    dt: f32,
    chunk_seed: u64,
) -> ChunkOutcome {
    let mut survivors = Vec::with_capacity(neutrons.len());
    let mut fissions = Vec::new();

    for neutron in neutrons {
        let jitter_seed = neutron.id ^ tick.rotate_left(7) ^ chunk_seed.rotate_left(13);
        let direction_noise = Vec2::new(
            random_signed(jitter_seed ^ 0x11),
            random_signed(jitter_seed ^ 0x22),
        ) * 0.2;
        let direction = (neutron.velocity.normalize_or_zero() + direction_noise).normalize_or_zero();
        let speed = config.neutron_speed * (0.88 + random01(jitter_seed ^ 0x33) * 0.32);
        let mut velocity = if direction.length_squared() > 0.0 {
            direction * speed
        } else {
            Vec2::new(speed, 0.0)
        };
        let mut position = neutron.position + velocity * dt;

        if position.x <= 0.0 || position.x >= config.width {
            velocity.x *= -1.0;
            position.x = position.x.clamp(0.0, config.width);
        }
        if position.y <= 0.0 || position.y >= config.height {
            velocity.y *= -1.0;
            position.y = position.y.clamp(0.0, config.height);
        }

        if rod.absorbs(position, config.height) {
            let absorption_chance = (0.68 + rod.depth * 0.28).clamp(0.0, 0.98);
            if random01(jitter_seed ^ 0x44) < absorption_chance {
                continue;
            }
        }

        if let Some(cell_idx) = cell_index(position, config) {
            let atom = fuel[cell_idx];
            if atom.active {
                let suppression = rod.suppression_at(position, config.height);
                let reactivity = config.base_fission_probability
                    * atom.enrichment
                    * atom.fissile_mass.max(0.2)
                    * atom.stability.max(0.2)
                    * (1.0 - suppression);
                if random01(jitter_seed ^ 0x55) < reactivity.clamp(0.01, 0.92) {
                    fissions.push(FissionEvent {
                        cell_idx,
                        position,
                        velocity,
                        source_id: neutron.id,
                        energy: neutron.energy,
                    });
                    continue;
                }
            }
        }

        let age = neutron.age + dt;
        if age <= config.neutron_lifetime {
            survivors.push(NeutronParticle {
                id: neutron.id,
                position,
                velocity,
                energy: (neutron.energy * 0.998).max(0.2),
                age,
            });
        }
    }

    ChunkOutcome { survivors, fissions }
}

fn apply_fission_events(state: &mut SimulationState, config: SimulationConfig, events: Vec<FissionEvent>) {
    let mut claimed = vec![false; state.fuel.len()];

    for event in events {
        if state.neutrons.len() >= config.max_neutrons {
            break;
        }

        if claimed[event.cell_idx] {
            continue;
        }
        claimed[event.cell_idx] = true;

        let atom = &mut state.fuel[event.cell_idx];
        if !atom.active {
            continue;
        }

        let released_mass = atom.fissile_mass * 0.58;
        atom.fissile_mass *= 0.42;
        atom.stability *= 0.96;
        atom.active = atom.fissile_mass > 0.08;

        state.last_fissions += 1;
        state.heat[event.cell_idx] += 1.4 + released_mass * 1.1 + event.energy * 0.25;

        let base_angle = event.velocity.y.atan2(event.velocity.x);
        for branch in 0..2 {
            let angle = base_angle + (branch as f32 * std::f32::consts::PI) + random_signed(event.source_id ^ branch as u64) * 0.32;
            state.fragments.push(FissionFragment {
                position: event.position,
                velocity: Vec2::from_angle(angle) * (config.neutron_speed * 0.45),
                energy: 1.0 + released_mass,
                life: FRAGMENT_LIFETIME,
            });
        }

        let neutron_release = 2 + usize::from(random01(event.source_id ^ state.tick) > 0.58);
        for branch in 0..neutron_release {
            if state.neutrons.len() >= config.max_neutrons {
                break;
            }

            let spread = TAU / neutron_release as f32;
            let angle = base_angle + spread * branch as f32 + random_signed((event.source_id << 3) ^ branch as u64) * 0.35;
            let position = event.position + Vec2::from_angle(angle) * (config.width / config.grid_width as f32 * 0.2);
            let velocity = Vec2::from_angle(angle) * (config.neutron_speed * (1.05 + 0.15 * branch as f32));
            let neutron = state.spawn_neutron(position, velocity, 1.25 + released_mass);
            state.neutrons.push(neutron);
        }
    }
}

fn build_frame(state: &SimulationState, config: SimulationConfig, sample_limit: usize) -> SimulationFrame {
    let total_particles = state.neutrons.len() + state.fragments.len();
    let sample_limit = sample_limit.max(1);
    let neutron_budget = sample_limit.saturating_mul(4) / 5;
    let neutron_stride = if state.neutrons.len() <= neutron_budget {
        1
    } else {
        (state.neutrons.len() + neutron_budget - 1) / neutron_budget
    };
    let mut sample_x = Vec::with_capacity(sample_limit.min(total_particles));
    let mut sample_y = Vec::with_capacity(sample_limit.min(total_particles));
    let mut sample_kind = Vec::with_capacity(sample_limit.min(total_particles));
    let mut sample_energy = Vec::with_capacity(sample_limit.min(total_particles));

    for neutron in state.neutrons.iter().step_by(neutron_stride) {
        if sample_x.len() >= sample_limit {
            break;
        }
        sample_x.push(neutron.position.x);
        sample_y.push(neutron.position.y);
        sample_kind.push(0);
        sample_energy.push(neutron.energy);
    }

    let remaining_budget = sample_limit.saturating_sub(sample_x.len()).max(1);
    let fragment_stride = if state.fragments.len() <= remaining_budget {
        1
    } else {
        (state.fragments.len() + remaining_budget - 1) / remaining_budget
    };

    for fragment in state.fragments.iter().step_by(fragment_stride) {
        if sample_x.len() >= sample_limit {
            break;
        }
        sample_x.push(fragment.position.x);
        sample_y.push(fragment.position.y);
        sample_kind.push(1);
        sample_energy.push(fragment.energy);
    }

    SimulationFrame {
        tick: state.tick,
        neutron_population: state.neutrons.len(),
        average_heat: average_heat(&state.heat),
        reaction_rate: *state.history_reaction_rate.last().unwrap_or(&0.0),
        rod_depth: state.rod.depth,
        rod_x: state.rod.x,
        rod_width: state.rod.width,
        core_width: config.width,
        core_height: config.height,
        grid_width: config.grid_width,
        grid_height: config.grid_height,
        sample_x,
        sample_y,
        sample_kind,
        sample_energy,
        density_map: state.density.clone(),
        heat_map: state.heat.clone(),
        history_ticks: state.history_ticks.clone(),
        history_population: state.history_population.clone(),
        history_average_heat: state.history_average_heat.clone(),
        history_reaction_rate: state.history_reaction_rate.clone(),
    }
}

fn average_heat(heat: &[f32]) -> f32 {
    if heat.is_empty() {
        0.0
    } else {
        heat.iter().sum::<f32>() / heat.len() as f32
    }
}

fn cell_index(position: Vec2, config: SimulationConfig) -> Option<usize> {
    let gx = ((position.x / config.width) * config.grid_width as f32).floor() as isize;
    let gy = ((position.y / config.height) * config.grid_height as f32).floor() as isize;

    if gx < 0 || gy < 0 || gx >= config.grid_width as isize || gy >= config.grid_height as isize {
        return None;
    }

    Some(gy as usize * config.grid_width + gx as usize)
}

fn random01(seed: u64) -> f32 {
    let bits = splitmix64(seed) >> 40;
    bits as f32 / ((1u64 << 24) - 1) as f32
}

fn random_signed(seed: u64) -> f32 {
    random01(seed) * 2.0 - 1.0
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
