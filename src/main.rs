use std::collections::HashMap;

mod genome;
mod misc;
mod trainer;
use genome::{Gene, Genome, NodeType};
use trainer::Trainer;

fn main() {
    // Create test genome
    let genome = Genome::<Sensor, Output> {
        nodes: vec![
            NodeType::Sensor(Sensor::A),
            NodeType::Sensor(Sensor::B),
            NodeType::Hidden,
            NodeType::Output(Output),
        ],
        genes: vec![bgene(1, 3, 0.7), bgene(0, 2, 0.2), bgene(2, 3, -0.3)],
    };

    // Simplate
    // - Run user supplied code to give the sensors values
    // - Propagate sensor info through edges
    // - Assign output values
    let mut map = HashMap::new();
    map.insert(Sensor::A, 1.0);
    map.insert(Sensor::B, 1.0);

    let out = genome.simulate(map);
    dbg!(&out);
    println!("Fitness: {}", fitness(out));
}

fn fitness(out: HashMap<Output, f32>) -> f32 {
    let raw = out.get(&Output).unwrap();
    1. - raw
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum Sensor {
    A,
    B,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Output;

fn bgene(node_in: usize, node_out: usize, weight: f32) -> Gene {
    Gene {
        node_in,
        node_out,
        weight,
        enabled: true,
        innovation: 0,
    }
}
