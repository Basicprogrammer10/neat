mod genome;
use genome::{Gene, Genome, NodeType};

fn main() {
    // Create test genome
    let genome = Genome::<Sensor, Output> {
        nodes: vec![
            NodeType::Sensor(Sensor::A).into(),
            NodeType::Sensor(Sensor::B).into(),
            NodeType::Hidden.into(),
            NodeType::Output(Output).into(),
        ],
        genes: vec![bgene(1, 3, 0.7), bgene(0, 2, 0.2), bgene(2, 3, -0.3)],
    };

    // Simplate
    // - Run user supplied code to give the sensors values
    // - Propagate sensor info through edges
    // - Assign output values
}

enum Sensor {
    A,
    B,
}

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
