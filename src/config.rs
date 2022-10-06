pub struct Config {
    // == BASIC ==
    /// The size of the population
    pub population_size: usize,

    // == POPULATION INIT ==
    /// The chance of a node to have an edge on population init
    pub init_edge_chance: f32,

    // == COMPATIBILITY COEFFICIENTS ==
    // Referr to the NEAT documentation on these,,,
    pub compatibility_1: f32,
    pub compatibility_2: f32,
    pub compatibility_3: f32,
    pub compatibility_threshold: f32,

    // == MUTATION CHANCES ==
    /// The chance to mutate an edges weight
    pub mutate_weight: f32,
    /// The chance to reset an edges weight (if being mutated)
    pub mutate_weight_reset: f32,
    /// The chance to add a node to genome
    pub mutate_add_node: f32,
    /// The chance to add an edge to genome
    pub mutate_add_edge: f32,
    /// The number of attempts to make on creating a new edge
    pub mutate_add_edge_tries: usize,
}

// Default values stolen from the NEAT paper
impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 150,
            init_edge_chance: 0.75,
            compatibility_1: 1.0,
            compatibility_2: 1.0,
            compatibility_3: 0.4,
            compatibility_threshold: 3.0,
            mutate_weight: 0.8,
            mutate_weight_reset: 0.1,
            mutate_add_node: 0.03,
            mutate_add_edge: 0.05,
            mutate_add_edge_tries: 20,
        }
    }
}
