use std::borrow::Borrow;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use parking_lot::RwLock;
use rand::{thread_rng, Rng};

use crate::genome::NodeType;
use crate::{config::Config, genome::Genome};

pub struct Trainer {
    // == INFO ==
    inputs: usize,
    outputs: usize,

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

    pub fn new_innovation(&self) -> usize {
        self.innovation.fetch_add(1, Ordering::AcqRel)
    }

    pub fn populate(self: Arc<Self>) {
        let mut agents = self.agents.write();
        let mut base_nodes = Vec::with_capacity(self.inputs + self.outputs);
        base_nodes.extend([NodeType::Sensor].repeat(self.inputs));
        base_nodes.extend([NodeType::Output].repeat(self.outputs));

        for _ in agents.len()..self.config.population_size {
            agents.push(Genome::new(self.clone(), base_nodes.clone()))
        }
    }

    pub fn species_categorize(self: Arc<Self>) -> Vec<usize> {
        let mut rng = thread_rng();
        let working = self.agents.borrow().read().clone();
        let mut working = working.iter().enumerate().collect::<Vec<_>>();
        let mut species = self.species.borrow().write();
        let mut out = vec![0; working.len()];

        'l: while working.len() > 0 {
            // Get and remove random genome
            let (gnome_index, genome) = working.remove(rng.gen_range(0..working.len()));

            // Compare it to every current species
            for x in species.iter() {
                let distance = x.1.distance(self.clone(), &genome);
                if distance < self.config.compatibility_threshold {
                    out[gnome_index] = x.0;
                    continue 'l;
                }
            }

            // Create a new speciesf
            let new_index = species.last().map(|x| x.0 + 1).unwrap_or(0);
            species.push((new_index, genome.clone()));
            out[gnome_index] = new_index;
        }

        // Prune unused species
        for e in species.clone().iter() {
            if !out.contains(&e.0) {
                species.retain(|x| x.0 != e.0);
            }
        }

        out
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
    pub fn species_fitness(&self, species: &[usize], fitness: &[f32]) -> Vec<f32> {
        let agents = self.agents.borrow().read();
        let mut out = vec![0.0; agents.len()];

        // nf = f / [count of genomes in the same spesies]
        for (i, e) in fitness.iter().enumerate() {
            // let this_species = species.get(i).unwrap().0;
            let this_species = species[i];
            let count = species.iter().filter(|x| **x == this_species).count();
            out[i] = fitness[i] / count as f32;
        }

        out
    }

    pub fn mutate_population(self: Arc<Self>) {
        let mut agents = self.agents.write();
        let mut mutations = Vec::new();

        for i in agents.iter_mut() {
            *i = i.mutate(self.clone(), &mut mutations);
        }
    }

    // Removes the worst performing genomes
    pub fn execute(&self, fitness: &[f32]) {
        let mut agents = self.agents.write();
        let to_remove = (agents.len() as f32 * self.config.population_kill_percent) as usize;
        let mut agent_fitness = fitness.iter().enumerate().collect::<Vec<_>>();
        agent_fitness.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        for (remove_index, (agent_index, _)) in agent_fitness.iter().take(to_remove).enumerate() {
            agents.remove(agent_index - remove_index);
        }
    }

    pub fn repopulate(&self, fitness: &[f32]) {
        let mut rng = thread_rng();
        let mut agents = self.agents.write();
        let mut new_agents = Vec::new();
        assert!(agents.len() > 1);

        while agents.len() + new_agents.len() < self.config.population_size {
            let i1 = rng.gen_range(0..agents.len());
            let i2 = rng.gen_range(0..agents.len());
            let g1 = &agents[i1];
            let g2 = &agents[i2];

            if i1 == i2 {
                continue;
            }

            new_agents.push(g1.crossover(g2, (fitness[i1], fitness[i2])));
        }

        agents.extend(new_agents);
    }
}

// fn get_pairs(&self) -> Vec<(Genome<S, O>, Genome<S, O>)> {
//     let mut agents = self.agents.read().to_vec();
//     let mut out = Vec::new();

//     while agents.len() > 1 {
//         let i1 = agents.remove(thread_rng().gen_range(0..agents.len()));
//         let i2 = agents.remove(thread_rng().gen_range(0..agents.len()));
//         out.push((i1, i2));
//     }

//     out
// }
