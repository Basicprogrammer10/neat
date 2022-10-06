use std::hash::Hash;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

use rand::{seq::IteratorRandom, thread_rng, Rng};

use crate::{
    misc::{sigmoid, SignString},
    trainer::Trainer,
};

#[derive(Clone)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
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

impl<S: Clone + Eq + Hash + Debug, O: Clone + Eq + Hash + Debug> Genome<S, O> {
    pub fn new(trainer: Arc<Trainer<S, O>>, io: Vec<NodeType<S, O>>) -> Self {
        let mut genes = Vec::new();
        for (i, e) in io.iter().enumerate() {
            match e {
                NodeType::Sensor(_) => {
                    if thread_rng().gen_bool(trainer.config.init_edge_chance.into()) {
                        // Get a random output node
                        let rand_out = io
                            .iter()
                            .enumerate()
                            .filter(|x| matches!(x.1, NodeType::Output(_)))
                            .choose(&mut thread_rng())
                            .expect("No Output Nodes");

                        // Make new geane
                        genes.push(Gene::random(trainer.new_innovation(), i, rand_out.0));
                    }
                }
                _ => continue,
            }
        }
        Self { nodes: io, genes }
    }

    pub fn distance(&self, other: &Self) {
        // δ = (c1 * E / N) + (c2 * D / N) + c3 * W
        // E: Excess geanes
        // D: Disjoint geanes
        // W: Weight diffrence between avraged
        // N: Geanes in the larger genome (normalised)
    }

    /// Use https://mermaid.live to render debug output
    pub fn debug(&self) -> String {
        let mut out = Vec::new();
        let mut remaining_nodes = self.nodes.clone();

        for i in self.genes.iter() {
            let node_in = &self.nodes[i.node_in];
            let node_out = &self.nodes[i.node_out];

            out.push(format!(
                r#"{}("{:?}") -{t} {} {t}-> {}["{:?}"]"#,
                i.node_in,
                node_in,
                i.weight.sign_str(),
                i.node_out,
                node_out,
                t = if i.enabled { "-" } else { "." }
            ));

            remaining_nodes.retain(|x| x != node_in);
            remaining_nodes.retain(|x| x != node_out);
        }

        for (i, e) in remaining_nodes.iter().enumerate() {
            match e {
                NodeType::Sensor(_) => out.push(format!(r#"unused-{}("{:?}")"#, i, e)),
                NodeType::Output(_) => out.push(format!(r#"unused-{}["{:?}"]"#, i, e)),
                _ => panic!(),
            }
        }

        out.join("\n")
    }

    // Checks if a edge from a -> b would cause a loop in the nural network
    // Use depth first search,,, smh
    pub fn would_be_recursive(&self, a: usize, b: usize) -> bool {
        // Make new network with a => b
        if a == b {
            return true;
        }

        self.genes
            .iter()
            .filter(|x| x.enabled)
            .any(|x| x.node_in == b && self.would_be_recursive(a, x.node_out))
    }

    pub fn mutate(&self, trainer: Arc<Trainer<S, O>>) -> Self {
        let mut rng = thread_rng();
        let mut this = self.clone();
        let nodes = this.nodes.len();

        // Mutate Weights
        for i in &mut this.genes {
            if rng.gen_bool(trainer.config.mutate_weight.into()) {
                if rng.gen_bool(trainer.config.mutate_weight.into()) {
                    i.weight = rng.gen_range(-1f32..=1f32);
                    continue;
                }
                i.weight *= rng.gen::<f32>()
            }
        }

        // Add Edge
        // TODO: Needs optimization for large networks (thousands of edges)
        if rng.gen_bool(trainer.config.mutate_add_edge.into()) {
            for _ in 0..trainer.config.mutate_add_edge_tries {
                // Genarate Indexes

                let a = rng.gen_range(0..nodes);
                let b = rng.gen_range(0..nodes);

                // Verify Indexes
                // Make sure not pointing to the same node twice, going in order of sensor => (hidden) => output
                // not the other way around and the connection would not make a recursive connection
                if a == b
                    || this.genes.iter().any(|x| x.connects(a, b))
                    || matches!(this.nodes[a], NodeType::Output(_))
                    || matches!(this.nodes[b], NodeType::Sensor(_))
                    || this.would_be_recursive(a, b)
                {
                    continue;
                }

                this.genes
                    .push(Gene::random(trainer.new_innovation(), a, b));
                break;
            }
        }

        // Add Node
        if !this.genes.is_empty() && rng.gen_bool(trainer.config.mutate_add_node.into()) {
            let gene = this
                .genes
                .iter_mut()
                .filter(|x| x.enabled)
                .choose(&mut rng)
                .unwrap();
            let old_node_from = gene.node_in;
            let old_node_to = gene.node_out;

            gene.enabled = false;
            this.nodes.push(NodeType::Hidden);
            this.genes.push(Gene {
                node_in: old_node_from,
                node_out: nodes,
                weight: 1.0,
                enabled: true,
                innovation: trainer.new_innovation(),
            });
            this.genes
                .push(Gene::random(trainer.new_innovation(), nodes, old_node_to));
        }

        this
    }

    pub fn simulate(&self, sensors: HashMap<S, f32>) -> HashMap<O, f32> {
        let mut out = HashMap::new();
        let node_tester = NodeTester::from_genome(self, sensors);

        for (i, e) in self.nodes.iter().enumerate() {
            match e {
                NodeType::Output(o) => {
                    out.insert(o.clone(), sigmoid(node_tester.clone().prop(i)));
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
            let val = match &ref_node.node {
                NodeType::Sensor(_) => {
                    // println!("S] {} {}=> {}", i.node_in, i.weight.sign_str(), to);
                    ref_node.value
                }
                _ => {
                    // println!("R] {} {}=> {}", i.node_in, i.weight.sign_str(), to);
                    self.prop(i.node_in)
                }
            };
            out += val * i.weight;
        }

        out
    }
}

impl Gene {
    fn random(innovation: usize, from: usize, to: usize) -> Self {
        Self {
            node_in: from,
            node_out: to,
            weight: thread_rng().gen_range(-1f32..=1f32),
            enabled: true,
            innovation,
        }
    }

    fn connects(&self, a: usize, b: usize) -> bool {
        (self.node_in == a && self.node_out == b) || (self.node_in == b && self.node_out == a)
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
