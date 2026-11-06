use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, Neg, Sub};

use crate::rng::Mt19937_64;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The element type of a vector: `f32` or `f64`.
pub trait Float:
    sealed::Sealed
    + Copy
    + Debug
    + Display
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + DivAssign
{
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    const MAX: Self;

    fn from_usize(value: usize) -> Self;
    fn from_u8(value: u8) -> Self;
    fn to_f64(self) -> f64;
    /// Truncates to `u8`, saturating at 0 and 255; NaN becomes 0.
    fn to_u8(self) -> u8;
    fn floor(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn total_cmp(&self, other: &Self) -> Ordering;
    /// One uniform draw from `[0, 1)`, as `std::generate_canonical` in libstdc++.
    fn canonical(rng: &mut Mt19937_64) -> Self;
}

macro_rules! impl_float {
    ($t:ident) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const HALF: Self = 0.5;
            const MAX: Self = $t::MAX;

            fn from_usize(value: usize) -> Self {
                value as $t
            }
            fn from_u8(value: u8) -> Self {
                value as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn to_u8(self) -> u8 {
                self as u8
            }
            fn floor(self) -> Self {
                $t::floor(self)
            }
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            fn ln(self) -> Self {
                $t::ln(self)
            }
            fn total_cmp(&self, other: &Self) -> Ordering {
                $t::total_cmp(self, other)
            }
            fn canonical(rng: &mut Mt19937_64) -> Self {
                const TWO_POW_64: $t = 18446744073709551616.0;
                let value = rng.next_u64() as $t / TWO_POW_64;
                if value >= 1.0 {
                    1.0 - $t::EPSILON / 2.0 // largest value below 1
                } else {
                    value
                }
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

/// Squared Euclidean distance. Panics if the lengths differ.
pub fn l2_distance<F: Float>(v1: &[F], v2: &[F]) -> F {
    assert_eq!(v1.len(), v2.len(), "vectors must have the same length");
    let mut distance = F::ZERO;
    for (&a, &b) in v1.iter().zip(v2) {
        let diff = a - b;
        distance += diff * diff;
    }
    distance
}

/// Indices and values of the `n` smallest distances, ascending. Ties go to the
/// lower index, and NaNs sort last.
pub fn top_n<F: Float>(distances: &[F], n: usize) -> Vec<(usize, F)> {
    let mut results: Vec<(usize, F)> = distances.iter().copied().enumerate().collect();
    let cmp = |a: &(usize, F), b: &(usize, F)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
    if n == 0 {
        results.clear();
    } else if n < results.len() {
        results.select_nth_unstable_by(n - 1, cmp);
        results.truncate(n);
    }
    results.sort_unstable_by(cmp);
    results
}
