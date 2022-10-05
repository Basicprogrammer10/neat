use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use parking_lot::RwLock;

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
}
