use std::fmt::Debug;
use std::sync::Arc;
use std::{collections::HashMap, hash::Hash};

mod config;
mod genome;
mod misc;
mod trainer;
use genome::{Genome, NodeType};

use crate::trainer::Trainer;

fn main() {
    let trainer = Arc::new(Trainer::<Sensor, Output>::new());
    trainer.clone().populate(vec![
        NodeType::Sensor(Sensor::A),
        NodeType::Sensor(Sensor::B),
        NodeType::Output(Output),
    ]);

    for gen in 0..200 {
        let species = trainer.clone().species_categorize();
        let fitness = trainer.species_fitness(&species, &trainer.fitness(fit));

        trainer.repopulate(&fitness);
        trainer.clone().mutate_population();

        let maxfit = fitness.iter().fold(f32::MIN, |x, i| x.max(*i));
        println!("[*] GEN: {gen} MAXFIT: {maxfit}")
    }
}

fn fit(_: usize, g: &Genome<Sensor, Output>) -> f32 {
    let mut sum = 0.0;
    let mut map = HashMap::new();

    for i in [[false, false], [false, true], [true, false], [true, true]] {
        map.insert(Sensor::A, i[0] as usize as f32);
        map.insert(Sensor::B, i[1] as usize as f32);

        let real = (i[0] ^ i[1]) as usize as f32;
        let got = *g.simulate(&map).get(&Output).unwrap();
        sum += (real - got).abs();
    }

    4.0 - sum
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum Sensor {
    A,
    B,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Output;

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
    - Matching neanes are randomly selected from each parent whule the extra geanes are pulled from the more fit parent
    * The distance function says that `distance = (E * c1 / N) + (D * c2 / N) + c3 * <AVG WEIGHT>` where E is the count of excess geanes
      D is the count of disjoint geanes N is the count of geanes in the larger genome and the coeffecents aredefined in a config (Referer to page 109 - 110)

- Selection
* Debug weird freezes when doing tens of thousands of mutations
- While less than 15 genes bias add node operation to older genes
- When repopulating, remove the worse proforming genomes first. Then crossover.
- Look into nuron bias
- Past mutations
- Make system work with inout node counts not vecs of them
- Dont store nodes as real objects just counts?
*/

// TEST GENOME
// let genome = Genome::<Sensor, Output> {
//     nodes: vec![
//         NodeType::Sensor(Sensor::A),
//         NodeType::Sensor(Sensor::B),
//         NodeType::Hidden,
//         NodeType::Output(Output),
//     ],
//     genes: vec![bgene(1, 3, 0.7), bgene(0, 2, 0.2), bgene(2, 3, -0.3)],
// };
