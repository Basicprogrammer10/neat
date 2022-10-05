pub struct Trainer<S, O> {
    supplier: Box<dyn Fn() -> Vec<(S, f32)> + Send + Sync + 'static>,
    fitness: Box<dyn Fn(Vec<(O, f32)>) -> f32 + Send + Sync + 'static>,
}

impl<S, O> Trainer<S, O> {
    pub fn new(
        supplier: impl Fn() -> Vec<(S, f32)> + Send + Sync + 'static,
        fitness: impl Fn(Vec<(O, f32)>) -> f32 + Send + Sync + 'static,
    ) -> Self {
        Self {
            supplier: Box::new(supplier),
            fitness: Box::new(fitness),
        }
    }
}
