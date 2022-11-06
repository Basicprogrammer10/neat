use std::sync::atomic::Ordering;

use crate::{genome::Genome, innovation::SpecieCount};

pub struct Specie {
    /// Id of the spesie
    pub id: SpecieCount,

    /// The genome that classifies the spesies
    pub owner: Genome,
    /// The number of agents in the species
    pub count: usize,
    /// The genaration at which the spesies was created
    _age: usize,
    /// The last fitness of the spesies
    fitness: Option<f32>,
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
                _age: owner.trainer.gen.load(Ordering::Acquire),
                owner,
                count: 0,
                fitness: None,
                stagnant: 0,
            },
        )
    }

    /// Kill a set percent of the population
    pub fn kill(&self) {
        let mut species = self.this_species();
        let to_remove =
            (species.len() as f32 * self.owner.trainer.config.population_kill_percent) as usize;
        species.sort_by(|a, b| a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap());
        let remove = species
            .iter()
            .take(to_remove)
            .map(|x| x.id)
            .collect::<Vec<_>>();
        self.owner
            .trainer
            .agents
            .write()
            .retain(|x| !remove.contains(&x.id));
    }

    /// Update a species fitness
    pub fn update_fitness(&mut self) {
        let species = self.this_species();
        let len = species.len();
        let mut sum = 0.0;

        for i in species {
            sum += i.fitness.unwrap();
        }

        let mut fitness = sum / len as f32;
        if fitness.is_nan() {
            fitness = 0.0;
        }

        if fitness <= self.fitness.unwrap_or(0.0) {
            self.stagnant += 1;
        } else {
            self.stagnant = 0;
        }

        self.fitness = Some(fitness);
    }

    /// Gets the number of agents within the specie
    /// This does rely on the `species_categorize` function being called before
    pub fn count(&self) -> usize {
        self.this_species().len()
    }

    fn this_species(&self) -> Vec<Genome> {
        self.owner
            .trainer
            .agents
            .read()
            .iter()
            .filter(|x| x.species.unwrap() == self.id)
            .cloned()
            .collect()
    }
}
