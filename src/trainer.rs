use std::borrow::Borrow;
use std::mem;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use parking_lot::RwLock;
use rand::{thread_rng, Rng};

use crate::{config::Config, genome::Genome};

pub struct Trainer {
    // == INFO ==
    pub inputs: usize,
    pub outputs: usize,

    // == GENOME ==
    pub agents: RwLock<Vec<Genome>>,
    /// Species ID, Case 0 Genome
    pub species: RwLock<Vec<(usize, Genome)>>,
    innovation: AtomicUsize,

    // == SIMULATION ==
    pub config: Config,
}

impl Trainer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            inputs,
            outputs,
            agents: RwLock::new(Vec::new()),
            species: RwLock::new(Vec::new()),
            innovation: AtomicUsize::new(0),
            config: Config::default(),
        }
    }

    pub(crate) fn new_innovation(&self) -> usize {
        self.innovation.fetch_add(1, Ordering::AcqRel)
    }

    /// Create the innitial population
    pub fn populate(self: Arc<Self>) -> Arc<Self> {
        let return_self = self.clone();
        let mut agents = self.agents.write();

        for _ in agents.len()..self.config.population_size {
            agents.push(Genome::new(self.clone()))
        }

        return_self
    }

    pub fn species_categorize(&self) {
        let mut rng = thread_rng();
        let mut agents = self.agents.borrow().write();
        let mut species = self.species.borrow().write();
        let mut working = agents.clone();
        let mut used_species = Vec::new();

        'l: while !working.is_empty() {
            // Get and remove random genome
            let genome_index = rng.gen_range(0..working.len());
            let genome = working.remove(genome_index);

            // Compare it to every current species
            for x in species.iter() {
                let distance = x.1.distance(&genome);
                if distance < self.config.compatibility_threshold {
                    agents[genome_index].species = Some(x.0);
                    used_species.push(x.0);
                    continue 'l;
                }
            }

            // Create a new species
            let new_index = species.last().map(|x| x.0 + 1).unwrap_or(0);
            species.push((new_index, genome.clone()));
            agents[genome_index].species = Some(new_index);
            used_species.push(new_index);
        }

        // Prune unused species
        species.retain(|x| used_species.contains(&x.0));
    }

    pub fn fitness(&self, fitness: impl Fn(usize, &Genome) -> f32) -> Vec<f32> {
        let agents = self.agents.borrow().read();
        agents
            .iter()
            .enumerate()
            .map(|(i, e)| (fitness)(i, e))
            .collect::<Vec<_>>()
    }

    /// Modifies a genome's fitness by the population of its spesies
    pub fn species_fitness(&self, fitness: &[f32]) -> Vec<f32> {
        let agents = self.agents.borrow().read();
        let mut out = Vec::with_capacity(agents.len());

        // nf = f / [count of genomes in the same spesies]
        for (i, e) in fitness.iter().enumerate() {
            let this_species = agents[i].species;
            let count = agents.iter().filter(|x| x.species == this_species).count();
            out.push(e / count as f32);
        }

        out
    }

    pub fn mutate_population(&self) {
        let mut agents = self.agents.write();
        let mut mutations = Vec::new();

        for i in agents.iter_mut() {
            *i = i.mutate(&mut mutations);
        }
    }

    // Removes the worst performing genomes
    pub fn execute(&self, fitness: &[f32]) {
        let mut agents = self.agents.write();
        let remove_count = (agents.len() as f32 * self.config.population_kill_percent) as usize;
        let mut agent_fitness = fitness.iter().enumerate().collect::<Vec<_>>();
        agent_fitness.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let to_remove = agent_fitness
            .iter()
            .take(remove_count)
            .map(|x| agents[x.0].id)
            .collect::<Vec<_>>();

        agents.retain(|x| !to_remove.contains(&x.id));
    }

    pub fn repopulate(&self, fitness: &[f32]) {
        let mut rng = thread_rng();
        let mut agents = self.agents.write();
        let mut new_agents = Vec::new();
        debug_assert!(agents.len() > 1);

        while new_agents.len() < self.config.population_size {
            let i1 = rng.gen_range(0..agents.len());
            let i2 = rng.gen_range(0..agents.len());
            let g1 = &agents[i1];
            let g2 = &agents[i2];

            if i1 == i2 {
                continue;
            }

            // let new = g1.crossover(g2, (fitness[i1], fitness[i2]));
            let mut new = g1.clone();
            new.id = self.new_innovation();
            if new.is_recursive() {
                continue;
            }

            new_agents.push(new);
        }

        mem::swap(&mut *agents, &mut new_agents);
        debug_assert_eq!(agents.len(), self.config.population_size);
    }
}
