use std::sync::Arc;

mod config;
mod genome;
mod misc;
mod trainer;
use genome::Genome;

use crate::trainer::Trainer;

fn main() {
    // Create a new trainer with 2 inputs and 1 output
    // Then populate it
    let trainer = Arc::new(Trainer::new(2, 1)).populate();
    let mut best = None;

    // Evolve for 200 genarations
    for gen in 1..=140 {
        // Catagorize the species
        trainer.species_categorize();
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
        println!("[*] GEN: {gen} | MAXFIT: {maxfit:.2}");

        trainer.execute(&fitness);
        trainer.repopulate(&fitness);
        trainer.mutate_population();
    }

    println!("{}", best.unwrap().debug());
}

// Define an XoR fitness function
fn fit(_: usize, g: &Genome) -> f32 {
    let mut sum = 0.0;

    for i in [[false, false], [false, true], [true, false], [true, true]] {
        let inp = [i[0] as usize as f32, i[1] as usize as f32];
        let real = (i[0] ^ i[1]) as usize as f32;
        let got = g.simulate(&inp)[0];
        sum += (real - got).abs();
    }

    4.0 - sum
}

/*
== Example Implamentation =
- https://github.com/yaricom/goNEAT

== TODO ==

* New genomes start with 0 hidden nodes
* The types of mutations are as follows:
    * Weight Mutations (Same as normal GAs)
      Each edge has a chace for meing mutated or not
    * Structure Mutations (2 types)
      * Node additions: An edge is selected and disabled. A node is then inserted and two new edged
        are created. The one preceding the node is givin a weight of 1.0 and the one following the new node
        inhearits the old edges weight. The new geanes each net new innovations numbers.
      * Edge additions: A new edge with a random weight is added between two unconnected nodes
- Crossover
    * Matching neanes are randomly selected from each parent whule the extra geanes are pulled from the more fit parent
    * The distance function says that `distance = (E * c1 / N) + (D * c2 / N) + c3 * <AVG WEIGHT>` where E is the count of excess geanes
      D is the count of disjoint geanes N is the count of geanes in the larger genome and the coeffecents aredefined in a config (Referer to page 109 - 110)

* Selection
* Debug weird freezes when doing tens of thousands of mutations
- While less than 15 genes bias add node operation to older genes
- When repopulating, remove the worse proforming genomes first. Then crossover.
- Look into nuron bias
- Past mutations
* Make system work with inout node counts not vecs of them
- Dont store nodes as real objects just counts?
*/
