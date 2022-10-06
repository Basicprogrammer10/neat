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
    pub agents: RwLock<Vec<Genome<S, O>>>,
    innovation: AtomicUsize,
    pub config: Config,
}

impl<S: Clone + Eq + Hash + Debug, O: Clone + Eq + Hash + Debug> Trainer<S, O> {
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(Vec::new()),
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

    fn get_pairs(&self) -> Vec<(Genome<S, O>, Genome<S, O>)> {
        let mut agents = self.agents.read().to_vec();
        let mut out = Vec::new();

        while agents.len() > 1 {
            let i1 = agents.remove(thread_rng().gen_range(0..agents.len()));
            let i2 = agents.remove(thread_rng().gen_range(0..agents.len()));
            out.push((i1, i2));
        }

        out
    }
}
