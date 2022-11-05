use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;
use std::{cell::RefCell, cmp::Ordering};

use bitvec::prelude::Lsb0;
use bitvec::vec::BitVec;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};

use crate::{
    misc::{sigmoid, SignString},
    trainer::Trainer,
};

#[derive(Clone)]
pub struct Genome {
    trainer: Arc<Trainer>,

    pub genes: Vec<Gene>,
    nodes: usize,

    pub id: usize,
    pub species: Option<usize>,
}

#[derive(Copy, Clone, Debug)]
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
    pub fn new(trainer: Arc<Trainer>) -> Self {
        let mut genes = Vec::new();

        for i in 0..trainer.inputs {
            for o in 0..trainer.outputs {
                // Make new gene
                genes.push(Gene::random(
                    trainer.new_innovation(),
                    i,
                    trainer.inputs + o,
                ));
            }
        }

        Self {
            id: trainer.new_innovation(),
            species: None,
            genes,
            nodes: trainer.inputs + trainer.outputs,
            trainer,
        }
    }

    pub fn classify_node(&self, id: usize) -> NodeType {
        if id < self.trainer.inputs {
            return NodeType::Sensor;
        }

        if id - self.trainer.inputs < self.trainer.outputs {
            return NodeType::Output;
        }

        NodeType::Hidden
    }

    // δ = (c1 * E / N) + (c2 * D / N) + c3 * W
    // E: Excess genes
    // D: Disjoint genes
    // W: Weight difference between averaged
    // N: Genes in the larger genome (normalized)
    pub fn distance(&self, other: &Self) -> f32 {
        // Gets all the innovations numbers of the 'other' genome
        let other_innovations = other.genes.iter().map(|x| x.innovation).collect::<Vec<_>>();
        let self_innovations = self.genes.iter().map(|x| x.innovation).collect::<Vec<_>>();
        let matching_innovations = self
            .genes
            .iter()
            .map(|x| x.innovation)
            .filter(|x| other_innovations.contains(x))
            .collect::<Vec<_>>();

        // Get the count of excess genes
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

        // Get the larger gene count (normalized)
        let mut n = self.genes.len().max(other.genes.len());
        if n < 20 {
            n = 1;
        }

        // Get average weight difference
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
        // further research needed
        if w.is_nan() {
            w = 0.0;
        }

        // Distance equation
        let c1 = self.trainer.config.excess_comp;
        let c2 = self.trainer.config.disjoint_comp;
        let c3 = self.trainer.config.weight_comp;
        let n = n as f32;

        (c1 * e as f32 / n) + (c2 * d as f32 / n) + c3 * w as f32
    }

    /// Use https://mermaid.live to render debug output
    pub fn debug(&self) -> String {
        let mut out = Vec::new();
        let mut remaining_nodes = (0..self.nodes).collect::<Vec<_>>();

        for i in self.genes.iter() {
            let node_in = self.classify_node(i.node_in);
            let node_out = self.classify_node(i.node_out);

            out.push(format!(
                r#"{}("{:?}") -{t} {} {t}-> {}["{:?}"]"#,
                i.node_in,
                node_in,
                i.weight.sign_str(),
                i.node_out,
                node_out,
                t = if i.enabled { "-" } else { "." }
            ));

            remaining_nodes.retain(|x| *x != i.node_in);
            remaining_nodes.retain(|x| *x != i.node_out);
        }

        for (i, e) in remaining_nodes.iter().enumerate() {
            out.push(format!(r#"unused-{}("{:?}")"#, i, self.classify_node(*e)));
        }

        out.join("\n")
    }

    // Checks if a edge from a -> b would cause a loop in the neural network
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

    pub fn is_recursive(&self) -> bool {
        for i in 0..self.trainer.inputs {
            let mut seen_nodes = BitVec::<usize, Lsb0>::repeat(false, self.nodes);
            seen_nodes.set(i, true);

            let rc = Rc::new(RefCell::new(seen_nodes));
            if is_recursive_checker(rc.clone(), &self.genes, i) {
                return true;
            }
        }

        false
    }

    pub fn mutate(&self, _past_mutations: &mut [(usize, (usize, usize))]) -> Self {
        let mut rng = thread_rng();
        let mut this = self.clone();

        // Mutate Weights
        for i in this.genes.iter_mut().filter(|x| x.enabled) {
            if rng.gen_bool(self.trainer.config.mutate_weight.into()) {
                if rng.gen_bool(self.trainer.config.mutate_weight.into()) {
                    i.weight = rng.gen_range(-1f32..=1f32);
                    continue;
                }
                i.weight *= rng.gen_range(-1f32..=1f32);
            }

            if rng.gen_bool(self.trainer.config.mutate_disable_edge.into()) {
                i.enabled = false;
            }
        }

        // Add Edge
        if rng.gen_bool(self.trainer.config.mutate_add_edge.into()) {
            for _ in 0..self.trainer.config.mutate_add_edge_tries {
                // generate indices

                let a = rng.gen_range(0..self.nodes);
                let b = rng.gen_range(0..self.nodes);

                // Verify Indexes
                // Make sure not pointing to the same node twice, going in order of sensor => (hidden) => output
                // not the other way around and the connection would not make a recursive connection
                if a == b
                    || this.genes.iter().any(|x| x.connects(a, b))
                    || this.classify_node(a) == NodeType::Output
                    || this.classify_node(b) == NodeType::Sensor
                    || this.would_be_recursive(a, b)
                {
                    continue;
                }

                this.genes
                    .push(Gene::random(self.trainer.new_innovation(), a, b));
                break;
            }
        }

        // Add Node
        // ==
        debug_assert!(!this.genes.is_empty());
        if rng.gen_bool(self.trainer.config.mutate_add_node.into()) {
            let gene = this
                .genes
                .iter_mut()
                .filter(|x| x.enabled)
                .choose(&mut rng)
                .unwrap();
            let old_node_from = gene.node_in;
            let old_node_to = gene.node_out;

            gene.enabled = false;
            this.genes.push(Gene {
                node_in: old_node_from,
                node_out: this.nodes,
                weight: 1.0,
                enabled: true,
                innovation: self.trainer.new_innovation(),
            });
            this.genes.push(Gene::random(
                self.trainer.new_innovation(),
                this.nodes,
                old_node_to,
            ));
            this.nodes += 1;
        }
        // ==

        this
    }

    pub fn crossover(&self, other: &Self, fitness: (f32, f32)) -> Self {
        let mut rng = thread_rng();
        let mut genes = Vec::with_capacity(self.genes.len().max(other.genes.len()));

        let (matching, self_genes, other_genes) = gene_diff(&self.genes, &other.genes);

        // Add matching
        for i in matching {
            let mut gene = *if rng.gen_bool(0.4) { i.0 } else { i.1 };
            if !gene.enabled && !rng.gen_bool(self.trainer.config.crossover_keep_disabled.into()) {
                gene.enabled = true;
            }

            genes.push(gene);
        }

        // Add nonmatching
        let fitter_nonmatching = match fitness.0.partial_cmp(&fitness.1).unwrap() {
            Ordering::Greater => self_genes,
            Ordering::Less => other_genes,
            _ => [self_genes, other_genes]
                .choose(&mut rng)
                .unwrap()
                .to_owned(),
        };
        genes.extend(fitter_nonmatching.iter().copied());

        Genome {
            trainer: self.trainer.clone(),
            id: self.trainer.new_innovation(),
            species: None,
            genes,
            nodes: self.nodes.max(other.nodes),
        }
    }

    pub fn simulate(&self, sensors: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.trainer.outputs);
        let node_tester = Rc::new(NodeTester::from_genome(self, sensors));

        for i in self.trainer.inputs..self.trainer.inputs + self.trainer.outputs {
            out.push(node_tester.clone().prop(i));
        }

        out
    }
}

impl NodeTester {
    fn from_genome(genome: &Genome, sensors: &[f32]) -> Self {
        let mut sensors = sensors.iter();
        Self {
            nodes: (0..genome.nodes)
                .map(|_| RefCell::new(sensors.next().copied()))
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
        let seen = nodes.borrow()[i.node_out];
        nodes.borrow_mut().set(i.node_out, true);
        if seen || is_recursive_checker(seen_nodes.clone(), genes, i.node_out) {
            return true;
        }
    }

    false
}

// -> (Matching Genes, A Genes, B Genes)
fn gene_diff<'a>(
    a: &'a [Gene],
    b: &'a [Gene],
) -> (Vec<(&'a Gene, &'a Gene)>, Vec<&'a Gene>, Vec<&'a Gene>) {
    let mut matching = Vec::new();
    let mut a_extra = Vec::new();
    let mut b_extra = b.iter().collect::<Vec<_>>();

    for i in a {
        let matched = match b.iter().find(|x| x.innovation == i.innovation) {
            Some(i) => i,
            None => {
                a_extra.push(i);
                continue;
            }
        };

        b_extra.retain(|x| x.innovation != i.innovation);
        matching.push((i, matched));
    }

    (matching, a_extra, b_extra)
}
