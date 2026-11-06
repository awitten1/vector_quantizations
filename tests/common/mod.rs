use vq::rng::{Mt19937_64, Normal};

/// n row-major dim-dimensional vectors.
pub struct Dataset {
    pub data: Vec<f32>,
    pub n: usize,
    pub dim: usize,
}

impl Dataset {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.dim..(i + 1) * self.dim]
    }
}

pub fn make_dataset(n: usize, dim: usize, seed: u64, stddev: f32) -> Dataset {
    let mut rng = Mt19937_64::new(seed);
    let mut dist = Normal::new(0.0, stddev);
    let data = (0..n * dim).map(|_| dist.sample(&mut rng)).collect();
    Dataset { data, n, dim }
}

pub fn random_doubles(count: usize, seed: u64) -> Vec<f64> {
    let mut rng = Mt19937_64::new(seed);
    let mut dist = Normal::standard();
    (0..count).map(|_| dist.sample(&mut rng)).collect()
}
