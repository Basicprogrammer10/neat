pub struct Config {
    // Basic
    pub population_size: usize,

    // Population Init
    pub init_edge_chance: f32
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            init_edge_chance: 0.75
        }
    }
}
