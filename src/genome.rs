use std::{collections::HashMap, hash::Hash};

use crate::misc::SignString;

pub struct Genome<S: Clone + Eq + Hash, O: Clone> {
    pub nodes: Vec<NodeType<S, O>>,
    pub genes: Vec<Gene>,
}

#[derive(Clone)]
pub struct Gene {
    pub node_in: usize,
    pub node_out: usize,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: usize,
}

#[derive(Clone)]
pub enum NodeType<S: Clone + Eq + Hash, O: Clone> {
    Sensor(S),
    Output(O),
    Hidden,
}

struct NodeTester<S: Clone + Eq + Hash, O: Clone> {
    pub nodes: Vec<TestNode<S, O>>,
    pub genes: Vec<Gene>,
}

struct TestNode<S: Clone + Eq + Hash, O: Clone> {
    node: NodeType<S, O>,
    value: f32,
}

impl<S: Clone + Eq + Hash, O: Clone + Eq + Hash> Genome<S, O> {
    pub fn simulate(&self, sensors: HashMap<S, f32>) -> HashMap<O, f32> {
        let mut node_tester = NodeTester::from_genome(self, sensors);
        for i in self.genes.iter().filter(|x| x.enabled) {
            node_tester.prop(i.node_in, i.node_out, i.weight);
        }

        let mut out = HashMap::new();
        for i in node_tester.nodes {
            match i.node {
                NodeType::Output(o) => out.insert(o, i.value),
                _ => continue,
            };
        }
        out
    }
}

impl<S: Clone + Eq + Hash, O: Clone> NodeTester<S, O> {
    fn from_genome(genome: &Genome<S, O>, sensors: HashMap<S, f32>) -> Self {
        Self {
            nodes: genome
                .nodes
                .iter()
                .cloned()
                .map(|x| match x {
                    NodeType::Sensor(ref s) => TestNode {
                        value: *sensors.get(s).unwrap(),
                        node: x,
                    },
                    _ => TestNode {
                        node: x,
                        value: 0.0,
                    },
                })
                .collect(),
            genes: genome.genes.clone(),
        }
    }

    fn prop(&mut self, from: usize, to: usize, weight: f32) {
        println!("{} {}=> {}", from, weight.sign_str(), to);

        // Update to node
        let from_val = self.nodes[from].value;
        self.nodes[to].value += from_val * weight;

        // Propagate changes
        for i in self
            .genes
            .clone()
            .iter()
            .filter(|x| x.enabled && x.node_in == to)
        {
            self.prop(to, i.node_out, i.weight);
        }
    }
}

impl<S: Clone + Eq + Hash, O: Clone> Into<TestNode<S, O>> for NodeType<S, O> {
    fn into(self) -> TestNode<S, O> {
        TestNode {
            node: self,
            value: 0.0,
        }
    }
}
