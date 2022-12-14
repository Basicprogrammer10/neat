pub trait SignString {
    fn sign_str(&self) -> String;
}

impl SignString for f32 {
    fn sign_str(&self) -> String {
        if self.is_sign_positive() {
            return format!("+{self}");
        }
        self.to_string()
    }
}

pub fn sigmoid(inp: f32) -> f32 {
    1.0 / (1.0 + (/*-4.9 */-1.0 * inp).exp())
}
