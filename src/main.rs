use std::sync::Arc;

mod config;
mod genome;
mod innovation;
mod misc;
mod species;
mod trainer;
use genome::Genome;

use crate::trainer::Trainer;

fn main() {
    // Create a new trainer with 2 inputs and 1 output
    // Then populate it
    let trainer = Arc::new(Trainer::new(3, 1)).populate();
    let mut best = None;

    // Evolve for 200 genarations
    for _ in 1..=200 {
        trainer.gen(fit);
    }

    let fitness = trainer.species_fitness(&trainer.fitness(fit));
    let maxfit = fitness.iter().fold(f32::MIN, |x, i| x.max(*i));
    best = Some(
        trainer.agents.read()[fitness
            .iter()
            .enumerate()
            .find(|x| *x.1 == maxfit)
            .unwrap()
            .0]
            .clone(),
    );

    println!("{}", best.unwrap().debug());
}

// Define an XoR fitness function
fn fit(_: usize, g: &Genome) -> f32 {
    let mut sum = 0.0;

    for i in [[false, false], [false, true], [true, false], [true, true]] {
        let inp = [1.0, i[0] as usize as f32, i[1] as usize as f32];
        let real = (i[0] ^ i[1]) as usize as f32;
        let got = g.simulate(&inp)[0];
        sum += (real - got).abs();
    }

    (4.0 - sum) / 4.0
}

/*
== Example Implementation =
- https://github.com/suhdonghwi/neat
- https://github.com/TLmaK0/rustneat
- https://github.com/yaricom/goNEAT

== TODO ==

* New genomes start with 0 hidden nodes
* The types of mutations are as follows:
    * Weight Mutations (Same as normal GAs)
      Each edge has a chance for being mutated or not
    * Structure Mutations (2 types)
      * Node additions: An edge is selected and disabled. A node is then inserted and two new edged
        are created. The one preceding the node is givin a weight of 1.0 and the one following the new node
        inherits the old edges weight. The new genes each net new innovations numbers.
      * Edge additions: A new edge with a random weight is added between two unconnected nodes
- Crossover
    * Matching genes are randomly selected from each parent while the extra genes are pulled from the more fit parent
    * The distance function says that `distance = (E * c1 / N) + (D * c2 / N) + c3 * <AVG WEIGHT>` where E is the count of excess genes
      D is the count of disjoint genes N is the count of genes in the larger genome and the coefficients are defined in a config (refer to page 109 - 110)

* Selection
* Debug weird freezes when doing tens of thousands of mutations
* While less than 15 genes bias add node operation to older genes
* When repopulating, remove the worse preforming genomes first. Then crossover.
* Look into neuron bias
* Past mutations
* Make system work with inout node counts not vecs of them
* Don't store nodes as real objects just counts?
* Fix crossover wackiness
* Spesies Struct system
* Genocide on the spesies
- "panic mode"
    - After not enough evolution only the top 2 spesies are allowed to repop
*/
