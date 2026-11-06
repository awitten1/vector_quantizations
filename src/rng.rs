//! Seeded random numbers: MT19937-64 and the distributions this crate draws
//! from. Each matches its libstdc++ counterpart bit for bit, so seeded results
//! line up with C++ code built against libstdc++. `Normal` calls the platform's
//! `ln`, so its exact bits can vary with the math library.

use crate::common::Float;

const STATE_SIZE: usize = 312;
const SHIFT_SIZE: usize = 156;
const XOR_MASK: u64 = 0xB502_6F5A_A966_19E9;
const UPPER_MASK: u64 = 0xFFFF_FFFF_8000_0000;
const LOWER_MASK: u64 = 0x0000_0000_7FFF_FFFF;

/// The 64-bit Mersenne Twister, equivalent to `std::mt19937_64`.
#[derive(Clone)]
pub struct Mt19937_64 {
    state: [u64; STATE_SIZE],
    index: usize,
}

impl Mt19937_64 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0; STATE_SIZE];
        state[0] = seed;
        for i in 1..STATE_SIZE {
            let prev = state[i - 1];
            state[i] = 6364136223846793005u64
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        Self { state, index: STATE_SIZE }
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.index == STATE_SIZE {
            self.refill();
        }
        let mut x = self.state[self.index];
        self.index += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^ (x >> 43)
    }

    fn refill(&mut self) {
        for i in 0..STATE_SIZE {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % STATE_SIZE] & LOWER_MASK);
            let twist = (y >> 1) ^ if y & 1 == 0 { 0 } else { XOR_MASK };
            self.state[i] = self.state[(i + SHIFT_SIZE) % STATE_SIZE] ^ twist;
        }
        self.index = 0;
    }
}

/// Uniform integer in `[0, range)`, as `std::uniform_int_distribution` (Lemire's
/// nearly divisionless method). `range` must be positive.
pub fn uniform_below(rng: &mut Mt19937_64, range: u64) -> u64 {
    assert!(range > 0, "range must be positive");
    let mut product = u128::from(rng.next_u64()) * u128::from(range);
    if (product as u64) < range {
        let threshold = range.wrapping_neg() % range;
        while (product as u64) < threshold {
            product = u128::from(rng.next_u64()) * u128::from(range);
        }
    }
    (product >> 64) as u64
}

/// Normal distribution, equivalent to `std::normal_distribution` (Marsaglia's
/// polar method, which produces values in pairs).
#[derive(Clone)]
pub struct Normal<F> {
    mean: F,
    stddev: F,
    saved: Option<F>,
}

impl<F: Float> Normal<F> {
    pub fn new(mean: F, stddev: F) -> Self {
        Self { mean, stddev, saved: None }
    }

    pub fn standard() -> Self {
        Self::new(F::ZERO, F::ONE)
    }

    pub fn sample(&mut self, rng: &mut Mt19937_64) -> F {
        let value = match self.saved.take() {
            Some(value) => value,
            None => {
                let two = F::ONE + F::ONE;
                let (x, y, r2) = loop {
                    let x = two * F::canonical(rng) - F::ONE;
                    let y = two * F::canonical(rng) - F::ONE;
                    let r2 = x * x + y * y;
                    if !(r2 > F::ONE || r2 == F::ZERO) {
                        break (x, y, r2);
                    }
                };
                let mult = (-two * r2.ln() / r2).sqrt();
                self.saved = Some(x * mult);
                y * mult
            }
        };
        value * self.stddev + self.mean
    }
}

/// Index drawn with probability proportional to its weight, equivalent to one
/// draw from a `std::discrete_distribution` built from `weights`. The weights
/// must have a positive sum.
pub fn discrete(weights: &[f64], rng: &mut Mt19937_64) -> usize {
    if weights.len() < 2 {
        return 0;
    }
    let total: f64 = weights.iter().fold(0.0, |sum, w| sum + w);
    let mut sum = 0.0;
    let mut cumulative: Vec<f64> = weights
        .iter()
        .map(|w| {
            sum += w / total;
            sum
        })
        .collect();
    *cumulative.last_mut().unwrap() = 1.0;
    let p = f64::canonical(rng);
    cumulative.partition_point(|&c| c < p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_64_matches_the_cpp_standard() {
        // [rand.predef]: the 10000th output of a default-constructed mt19937_64.
        let mut rng = Mt19937_64::new(5489);
        let last = (0..10000).map(|_| rng.next_u64()).last().unwrap();
        assert_eq!(last, 9981545732273789042);
    }
}
