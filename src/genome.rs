use std::{collections::HashMap, hash::Hash};

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

#[derive(Clone)]
struct NodeTester<S: Clone + Eq + Hash, O: Clone> {
    pub nodes: Vec<TestNode<S, O>>,
    pub genes: Vec<Gene>,
}

#[derive(Clone)]
struct TestNode<S: Clone + Eq + Hash, O: Clone> {
    node: NodeType<S, O>,
    value: f32,
}

impl<S: Clone + Eq + Hash, O: Clone + Eq + Hash> Genome<S, O> {
    pub fn simulate(&self, sensors: HashMap<S, f32>) -> HashMap<O, f32> {
        let mut out = HashMap::new();
        let node_tester = NodeTester::from_genome(self, sensors);

        for (i, e) in self.nodes.iter().enumerate() {
            match e {
                NodeType::Output(o) => {
                    out.insert(o.clone(), node_tester.clone().prop(i));
                }
                _ => continue,
            }
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

    fn prop(&mut self, to: usize) -> f32 {
        let mut out = 0.0;

        // Get nodes that connect to this one
        for i in self
            .genes
            .clone()
            .iter()
            .filter(|x| x.enabled && x.node_out == to)
        {
            // Check if the node this gene is refrencing is a sensor
            // If so add that to the out
            // Else recursively call prop function
            let ref_node = &self.nodes[i.node_in];
            out += match &ref_node.node {
                NodeType::Sensor(_) => ref_node.value,
                _ => self.prop(i.node_in),
            } * i.weight;
        }

        out
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
