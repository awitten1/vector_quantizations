//! Vector quantization: product quantization ([`ProductQuantizer`]) and 8-bit
//! scalar quantization ([`ScalarQuantizer`]), for `f32` and `f64` vectors.
//!
//! All batches of vectors are flat, row-major slices: vector `i` of a batch is
//! `data[i * dim..(i + 1) * dim]`.

mod common;
mod error;
mod pq;
pub mod rng;
mod sq;

pub use common::{Float, l2_distance, top_n};
pub use error::Error;
pub use pq::{DEFAULT_SEED, ProductQuantizer, kmeans_plus_plus};
pub use sq::ScalarQuantizer;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
