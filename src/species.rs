use std::sync::{atomic::Ordering, Arc};

use crate::{genome::Genome, innovation::SpecieCount, trainer::Trainer};

pub struct Specie {
    /// Id of the spesie
    pub id: SpecieCount,

    /// The genome that classifies the spesies
    pub owner: Genome,
    /// The genaration at which the spesies was created
    age: usize,
    /// The last fitness of the spesies
    fitness: Option<f64>,
    /// The number of genarations the fitness hasent gone up
    /// If it goes up this should be reset
    stagnant: usize,
}

impl Specie {
    /// -> (Specie ID, Specie)
    pub fn new(owner: Genome) -> (usize, Self) {
        let id = owner.trainer.innovator.new_specie();

        (
            id,
            Self {
                id,
                age: owner.trainer.gen.load(Ordering::Acquire),
                owner,
                fitness: None,
                stagnant: 0,
            },
        )
    }

    /// Gets the number of agents within the specie
    /// This does rely on the `species_categorize` function being called before
    pub fn count(&self) -> usize {
        let mut count = 0;

        for i in self.owner.trainer.agents.read().iter() {
            debug_assert!(i.species.is_some());
            count += (i.species.unwrap() == self.id) as usize;
        }

        count
    }
}
