use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use parking_lot::RwLock;
use rand::{thread_rng, Rng};

use crate::genome::NodeType;
use crate::{config::Config, genome::Genome};

pub struct Trainer<S: Clone + Eq + Hash + Debug, O: Clone + Debug> {
    // == GENOME ==
    pub agents: RwLock<Vec<Genome<S, O>>>,
    pub species: RwLock<Vec<(usize, Genome<S, O>)>>,
    innovation: AtomicUsize,

    // == SIMULATION ==
    pub config: Config,
}

impl<S: Clone + Eq + Hash + Debug, O: Clone + Eq + Hash + Debug> Trainer<S, O> {
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(Vec::new()),
            species: RwLock::new(Vec::new()),
            innovation: AtomicUsize::new(0),
            config: Config::default(),
        }
    }

    pub fn new_innovation(&self) -> usize {
        self.innovation.fetch_add(1, Ordering::AcqRel)
    }

    pub fn populate(self: Arc<Self>, io: Vec<NodeType<S, O>>) {
        let mut agents = self.agents.write();
        for _ in agents.len()..self.config.population_size {
            agents.push(Genome::new(self.clone(), io.clone()))
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

    /// Modifies a genome's fitness by the population of its spesies
    pub fn species_fitness(self: Arc<Self>, fitness: &[f32]) -> Vec<f32> {
        let agents = self.agents.borrow().read();
        let species = self.species.read();
        let mut out = vec![0.0; agents.len()];

        // nf = f / [count of genomes in the same spesies]
        for (i, e) in fitness.iter().enumerate() {
            let this_species = species.get(i).unwrap().0;
            let count = species.iter().filter(|x| x.0 == this_species).count();
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
}
