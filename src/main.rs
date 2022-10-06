use std::{collections::HashMap, sync::Arc};

mod config;
mod genome;
mod misc;
mod trainer;
use genome::NodeType;
use misc::sigmoid;

use crate::trainer::Trainer;

fn main() {
    let trainer = Arc::new(Trainer::<Sensor, Output>::new());
    trainer.clone().populate(vec![
        NodeType::Sensor(Sensor::A),
        NodeType::Sensor(Sensor::B),
        NodeType::Output(Output),
    ]);

    for i in trainer.agents.read().iter().take(1) {
        println!("graph TD\n{}", i.debug());

        let mut i = i.mutate(trainer.clone());
        for _ in 0..100 {
            i = i.mutate(trainer.clone());
        }

        println!("graph TD\n{}", i.debug());

        let mut map = HashMap::new();
        map.insert(Sensor::A, 1.0);
        map.insert(Sensor::B, 1.0);

        println!("SIMULATION");
        println!("\n> {}", i.simulate(map).get(&Output).unwrap());
    }

    // Simulate
    // let mut map = HashMap::new();
    // map.insert(Sensor::A, 1.0);
    // map.insert(Sensor::B, 1.0);

    // for (i, e) in trainer.agents.read().iter().enumerate() {
    //     println!("#{} {}", i, e.simulate(map.clone()).get(&Output).unwrap());
    // }
}

fn fitness(out: HashMap<Output, f32>) -> f32 {
    let raw = out.get(&Output).unwrap();
    sigmoid((1. - raw).abs()) //.powf(2.)
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
    - The distance function says that `distance = (E * c1 / N) + (D * c2 / N) + c3 * <AVG WEIGHT>` where E is the count of excess geanes
      D is the count of disjoint geanes N is the count of geanes in the larger genome and the coeffecents aredefined in a config (Referer to page 109 - 110)

- Selection
- Debug weird freezes when doing tens of thousands of mutations
- While less than 15 genes bias add node operation to older genes
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
