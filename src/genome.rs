pub struct Genome<S, O> {
    pub nodes: Vec<Node<S, O>>,
    pub genes: Vec<Gene>,
}

pub struct Node<S, O> {
    pub node_type: NodeType<S, O>,
}

pub struct Gene {
    pub node_in: usize,
    pub node_out: usize,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: usize,
}

pub enum NodeType<S, O> {
    Sensor(S),
    Output(O),
    Hidden,
}

impl<S, O> From<NodeType<S, O>> for Node<S, O> {
    fn from(nt: NodeType<S, O>) -> Self {
        Node { node_type: nt }
    }
}
