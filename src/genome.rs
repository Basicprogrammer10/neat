use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

use bitvec::prelude::Lsb0;
use bitvec::vec::BitVec;
use rand::{seq::IteratorRandom, thread_rng, Rng};

use crate::{
    misc::{sigmoid, SignString},
    trainer::Trainer,
};

#[derive(Clone)]
pub struct Genome {
    pub nodes: Vec<NodeType>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Sensor,
    Output,
    Hidden,
}

#[derive(Clone)]
struct NodeTester {
    pub nodes: Vec<RefCell<Option<f32>>>,
    pub genes: Vec<Gene>,
}

impl Genome {
    pub fn new(trainer: Arc<Trainer>, io: Vec<NodeType>) -> Self {
        let mut genes = Vec::new();
        for (i, e) in io.iter().enumerate() {
            match e {
                NodeType::Sensor => {
                    if thread_rng().gen_bool(trainer.config.init_edge_chance.into()) {
                        // Get a random output node
                        let rand_out = io
                            .iter()
                            .enumerate()
                            .filter(|x| *x.1 == NodeType::Output)
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

    // δ = (c1 * E / N) + (c2 * D / N) + c3 * W
    // E: Excess geanes
    // D: Disjoint geanes
    // W: Weight diffrence between avraged
    // N: Geanes in the larger genome (normalised)
    pub fn distance(&self, trainer: Arc<Trainer>, other: &Self) -> f32 {
        // Gets all the innovations numbers of the 'other' geanome
        let other_innovations = other.genes.iter().map(|x| x.innovation).collect::<Vec<_>>();
        let self_innovations = self.genes.iter().map(|x| x.innovation).collect::<Vec<_>>();
        let matching_innovations = self
            .genes
            .iter()
            .map(|x| x.innovation)
            .filter(|x| other_innovations.contains(x))
            .collect::<Vec<_>>();

        // Get the count of excess geanes
        let self_max_innovation = self.genes.iter().map(|x| x.innovation).sum::<usize>();
        let other_max_innovation = other_innovations.iter().sum::<usize>();
        let min_max_innovation = self_max_innovation.min(other_max_innovation);
        let e = if self_max_innovation > other_max_innovation {
            &self.genes
        } else {
            &other.genes
        }
        .iter()
        .filter(|x| x.innovation > min_max_innovation)
        .count();

        // Get the count of disjoint genes
        let mut disjoint_innovations = other_innovations;
        disjoint_innovations.extend(self_innovations);
        disjoint_innovations
            .retain(|x| *x < min_max_innovation && !matching_innovations.contains(x));
        let d = disjoint_innovations.len();

        // Get the larger gene count (normalised)
        let mut n = self.genes.len().max(other.genes.len());
        if n < 20 {
            n = 1;
        }

        // Get avrage weight diffrence
        let self_matching_weight = self
            .genes
            .iter()
            .filter(|x| matching_innovations.contains(&x.innovation))
            .map(|x| x.weight);
        let other_matching_weight = other
            .genes
            .iter()
            .filter(|x| matching_innovations.contains(&x.innovation))
            .map(|x| x.weight);
        let mut w = self_matching_weight
            .zip(other_matching_weight)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / matching_innovations.len() as f32;
        // Furthure Research Needed
        if w.is_nan() {
            w = 0.0;
        }

        // Distance Equastion
        let c1 = trainer.config.excess_comp;
        let c2 = trainer.config.disjoint_comp;
        let c3 = trainer.config.weight_comp;
        let n = n as f32;

        (c1 * e as f32 / n) + (c2 * d as f32 / n) + c3 * w as f32
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
                NodeType::Sensor => out.push(format!(r#"unused-{}("{:?}")"#, i, e)),
                NodeType::Output => out.push(format!(r#"unused-{}["{:?}"]"#, i, e)),
                _ => panic!(),
            }
        }

        out.join("\n")
    }

    // Checks if a edge from a -> b would cause a loop in the nural network
    fn would_be_recursive(&self, a: usize, b: usize) -> bool {
        let mut new = self.clone();
        new.genes.push(Gene {
            node_in: a,
            node_out: b,
            weight: 0.0,
            enabled: true,
            innovation: 0,
        });
        new.is_recursive()
    }

    fn is_recursive(&self) -> bool {
        for (i, _) in self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, x)| **x == NodeType::Sensor)
        {
            let mut seen_nodes = BitVec::<usize, Lsb0>::new();
            seen_nodes.extend([false].repeat(self.nodes.len()));
            seen_nodes.set(i, true);

            let rc = Rc::new(RefCell::new(seen_nodes));
            if is_recursive_checker(rc.clone(), &self.genes, i) {
                return true;
            }
        }

        false
    }

    pub fn mutate(
        &self,
        trainer: Arc<Trainer>,
        past_mutations: &mut Vec<(usize, (usize, usize))>,
    ) -> Self {
        let mut rng = thread_rng();
        let mut this = self.clone();
        let nodes = this.nodes.len();

        // Mutate Weights
        for i in this.genes.iter_mut().filter(|x| x.enabled) {
            if rng.gen_bool(trainer.config.mutate_weight.into()) {
                if rng.gen_bool(trainer.config.mutate_weight.into()) {
                    i.weight = rng.gen_range(-1f32..=1f32);
                    continue;
                }
                i.weight *= rng.gen::<f32>()
            }

            if rng.gen_bool(trainer.config.mutate_disable_edge.into()) {
                i.enabled = false;
            }
        }

        // Add Edge
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
                    || this.nodes[a] == NodeType::Output
                    || this.nodes[b] == NodeType::Sensor
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

    pub fn crossover(&self, other: &Self, fitness: (f32, f32)) -> Self {
        let mut rng = thread_rng();
        let mut genes = Vec::new();
        let mut nodes = Vec::new();

        for i in self.nodes.iter().chain(other.nodes.iter()) {
            if !nodes.contains(&i) || matches!(i, NodeType::Hidden) {
                nodes.push(i);
            }
        }

        let mut self_i = 0;
        let mut other_i = 0;
        let mut self_last = 0;
        let mut other_last = 0;

        while self_i < self.genes.len() && other_i < other.genes.len() {
            let self_innovation = self.genes[self_i].innovation;
            let other_innovation = other.genes[other_i].innovation;

            if self_innovation < other_innovation {
                self_i += 1;
                continue;
            }

            if self_innovation > other_innovation {
                other_i += 1;
                continue;
            }

            if fitness.0 > fitness.1 {
                genes.extend(self.genes[self_last..self_i].iter());
            } else if fitness.1 > fitness.0 {
                genes.extend(other.genes[other_last..other_i].iter());
            } else {
                if rng.gen_bool(0.5) {
                    genes.extend(self.genes[self_last..self_i].iter());
                } else {
                    genes.extend(other.genes[other_last..other_i].iter());
                }
            }

            if rng.gen_bool(0.5) {
                genes.push(&self.genes[self_i]);
            } else {
                genes.push(&other.genes[other_i]);
            }

            self_i += 1;
            other_i += 1;
            self_last = self_i;
            other_last = other_i;
        }

        if self.genes.len() > other.genes.len() {
            genes.extend(self.genes[self_last..].iter());
        } else if self.genes.len() < other.genes.len() {
            genes.extend(other.genes[other_last..].iter());
        } else if fitness.0 > fitness.1 {
            genes.extend(self.genes[self_last..].iter());
        } else if fitness.1 > fitness.0 {
            genes.extend(other.genes[other_last..].iter());
        } else if rng.gen_bool(0.5) {
            genes.extend(self.genes[self_last..].iter());
        } else {
            genes.extend(other.genes[other_last..].iter());
        }

        Genome {
            nodes: nodes.into_iter().cloned().collect(),
            genes: genes.into_iter().cloned().collect(),
        }
    }

    pub fn simulate(&self, sensors: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        let node_tester = Rc::new(NodeTester::from_genome(self, sensors));

        for (i, e) in self.nodes.iter().enumerate() {
            match e {
                NodeType::Output => out.push(node_tester.clone().prop(i)),
                _ => continue,
            }
        }

        out
    }
}

impl NodeTester {
    fn from_genome(genome: &Genome, sensors: &[f32]) -> Self {
        Self {
            nodes: genome
                .nodes
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    RefCell::new(match x {
                        NodeType::Sensor => Some(sensors[i]),
                        _ => None,
                    })
                })
                .collect(),
            genes: genome.genes.clone(),
        }
    }

    fn prop(self: Rc<Self>, to: usize) -> f32 {
        let mut out = 0.0;

        // Get nodes that connect to this one
        for i in self.genes.iter().filter(|x| x.enabled && x.node_out == to) {
            // Check if the node this gene is refrencing is a sensor
            // If so add that to the out
            // Else recursively call prop function
            let new_self = self.clone();
            let ref_node = &self.nodes[i.node_in];
            let val = ref_node
                .borrow()
                .unwrap_or_else(|| new_self.prop(i.node_in));
            out += val * i.weight;
        }

        sigmoid(out)
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

fn is_recursive_checker(seen_nodes: Rc<RefCell<BitVec>>, genes: &[Gene], index: usize) -> bool {
    let nodes = seen_nodes.clone();
    for i in genes.iter().filter(|x| x.enabled && x.node_in == index) {
        let seen = *nodes.borrow().get(i.node_out).unwrap();
        nodes.borrow_mut().set(i.node_out, true);
        if seen || is_recursive_checker(seen_nodes.clone(), genes, i.node_out) {
            return true;
        }
    }

    false
}
