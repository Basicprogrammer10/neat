use std::sync::atomic::{AtomicUsize, Ordering};

use ahash::{HashMap, HashMapExt};
use parking_lot::Mutex;

pub type EdgeCount = usize;
pub type SpecieCount = usize;
pub type GenomeCount = usize;

pub struct Innovations {
    // == Component Counts / IDs ==
    /// The id of a gene
    edge_count: AtomicUsize,
    /// The id of a specie
    specie_count: AtomicUsize,
    /// The id of a genome
    genome_count: AtomicUsize,

    // == Edge record ==
    /// Maps (a, b) -> edge index
    past_connection: Mutex<HashMap<(usize, usize), usize>>,
}

impl Innovations {
    pub fn new() -> Self {
        Self {
            edge_count: AtomicUsize::new(0),
            specie_count: AtomicUsize::new(0),
            genome_count: AtomicUsize::new(0),
            past_connection: Mutex::new(HashMap::new()),
        }
    }

    // == New Innovations ==
    pub fn new_edge(&self, x: (usize, usize)) -> EdgeCount {
        *self
            .past_connection
            .lock()
            .entry(x)
            .or_insert_with(|| self.edge_count.fetch_add(1, Ordering::AcqRel))
    }

    pub fn new_specie(&self) -> SpecieCount {
        self.specie_count.fetch_add(1, Ordering::AcqRel)
    }

    pub fn new_genome(&self) -> GenomeCount {
        self.genome_count.fetch_add(1, Ordering::AcqRel)
    }
}
