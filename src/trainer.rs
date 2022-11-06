use std::borrow::Borrow;
use std::mem;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

use parking_lot::RwLock;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use crate::innovation::Innovations;
use crate::species::Specie;
use crate::{config::Config, genome::Genome};

pub struct Trainer {
    // == INFO ==
    pub inputs: usize,
    pub outputs: usize,

    // == GENOME ==
    pub agents: RwLock<Vec<Genome>>,
    /// Species ID, Case 0 Genome
    pub species: RwLock<Vec<Specie>>,
    pub innovator: Innovations,

    // == SIMULATION ==
    pub config: Config,
    pub gen: AtomicUsize,
}

impl Trainer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            inputs,
            outputs,
            agents: RwLock::new(Vec::new()),
            species: RwLock::new(Vec::new()),
            innovator: Innovations::new(),
            config: Config::default(),
            gen: AtomicUsize::new(0),
        }
    }

    pub fn gen(&self, fit: impl Fn(usize, &Genome) -> f32) {
        let start = Instant::now();
        self.species_categorize();
        let fitness = self.species_fitness(&self.fitness(fit));
        let maxfit = fitness.iter().fold(f32::MIN, |x, i| x.max(*i));

        self.execute(&fitness);
        self.repopulate(&fitness);
        self.mutate_population();
        self.gen.fetch_add(1, Ordering::AcqRel);
        println!(
            "GEN: {:3} | MAXFIT: {:3.0}% | SPEC: {:2} | TIME: {}ms",
            self.gen.load(Ordering::Acquire),
            maxfit * 100.,
            self.species.read().len(),
            start.elapsed().as_millis()
        );
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
        let working = agents.clone();
        let mut working = working.iter().enumerate().collect::<Vec<_>>();
        let mut used_species = Vec::new();

        'l: while !working.is_empty() {
            // Get and remove random genome
            let (agent_index, genome) = working.remove(rng.gen_range(0..working.len()));

            // Compare it to every current species
            for x in species.iter() {
                let distance = x.owner.distance(genome);
                if distance < self.config.compatibility_threshold {
                    agents[agent_index].species = Some(x.id);
                    used_species.push(x.id);
                    continue 'l;
                }
            }

            // Create a new species
            let (new_index, specie) = Specie::new(genome.clone());
            species.push(specie);
            agents[agent_index].species = Some(new_index);
            used_species.push(new_index);
        }

        // Prune unused species
        species.retain(|x| used_species.contains(&x.id));

        debug_assert!(agents.iter().all(|x| x.species.is_some()));
    }

    // TODO: Hashmap?
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

        for i in agents.iter_mut() {
            *i = i.mutate();
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
            // Find random genome
            let i1 = rng.gen_range(0..agents.len());
            let g1 = &agents[i1];
            debug_assert!(g1.species.is_some());

            // Find another one within its species
            let matching_agents = agents
                .iter()
                .enumerate()
                .filter(|x| x.1.species == g1.species)
                .collect::<Vec<_>>();
            let (mut i2, mut g2) = matching_agents.choose(&mut rng).unwrap();

            if matching_agents.len() <= 1 {
                let index_agents = agents.iter().enumerate().collect::<Vec<_>>();
                let rand = index_agents.choose(&mut rng).unwrap();
                i2 = rand.0;
                g2 = rand.1;
            }

            if i1 == i2 {
                continue;
            }

            let mut tries = self.config.mutate_add_edge_tries;
            let mut new = None;
            while tries > 0 {
                new = Some(g1.crossover(g2, (fitness[i1], fitness[i2])));
                if new.as_ref().unwrap().is_recursive() {
                    tries -= 1;
                    continue;
                }

                break;
            }

            new_agents.push(match new {
                Some(i) => i,
                None => continue,
            });
        }

        mem::swap(&mut *agents, &mut new_agents);
        debug_assert_eq!(agents.len(), self.config.population_size);
    }
}
