pub struct Config {
    // == BASIC ==
    /// The size of the population
    pub population_size: usize,

    // == POPULATION  ==
    /// The chance of a node to have an edge on population init
    // pub init_edge_chance: f32,
    /// Percent of the popluation to eggstermanate before repopulation
    pub population_kill_percent: f32,

    // == COMPATIBILITY COEFFICIENTS ==
    pub excess_comp: f32,
    pub disjoint_comp: f32,
    pub weight_comp: f32,
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
    /// The chance to disable an edge
    pub mutate_disable_edge: f32,

    // == CROSSOVER CHANCES ==
    pub crossover_keep_disabled: f32,
    pub crossover_trys: usize,
}

// Default values stolen from the NEAT paper
impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 150,
            population_kill_percent: 0.20,
            excess_comp: 1.0,
            disjoint_comp: 1.0,
            weight_comp: 0.4,
            compatibility_threshold: 3.0,
            mutate_weight: 0.8,
            mutate_weight_reset: 0.1,
            mutate_add_node: 0.03,
            mutate_add_edge: 0.05,
            mutate_add_edge_tries: 20,
            mutate_disable_edge: 0.0,
            crossover_keep_disabled: 0.4,
            crossover_trys: 1,
        }
    }
}
