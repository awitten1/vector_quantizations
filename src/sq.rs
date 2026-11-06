use crate::common::{Float, top_n};
use crate::error::Error;

const LEVELS: usize = 256;
const BLOCK: usize = 16; // dimensions processed at a time when scanning

/// 8-bit scalar quantizer with a separate range per dimension, as in FAISS
/// ScalarQuantizer QT_8bit with RS_minmax. Training records each dimension's
/// `[min, max]` and splits it into 256 equal buckets; a value is stored as its
/// bucket index and reconstructed at the bucket center.
#[derive(Clone, Debug)]
pub struct ScalarQuantizer<F> {
    dim: usize,
    min: Vec<F>,  // per dimension
    step: Vec<F>, // per dimension: (max - min) / 256
}

impl<F: Float> ScalarQuantizer<F> {
    /// Trains on the contiguous `dim`-dimensional vectors in `data`.
    pub fn train(data: &[F], dim: usize) -> Result<Self, Error> {
        if dim == 0 {
            return Err(Error::InvalidDim);
        }
        if data.len() % dim != 0 {
            return Err(Error::DataLengthMismatch);
        }
        if data.is_empty() {
            return Err(Error::NoTrainingVectors);
        }
        let mut min = data[..dim].to_vec();
        let mut max = min.clone();
        for x in data.chunks_exact(dim).skip(1) {
            for d in 0..dim {
                if x[d] < min[d] {
                    min[d] = x[d];
                }
                if max[d] < x[d] {
                    max[d] = x[d];
                }
            }
        }
        let levels = F::from_usize(LEVELS);
        let step = max.iter().zip(&min).map(|(&hi, &lo)| (hi - lo) / levels).collect();
        Ok(Self { dim, min, step })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encodes one vector into `dim` codes.
    pub fn encode(&self, vec: &[F]) -> Vec<u8> {
        let mut code = vec![0; self.dim];
        self.encode_into(vec, &mut code);
        code
    }

    /// Encodes contiguous vectors into a flat array of `n * dim` codes. Panics
    /// if `data` is not a whole number of vectors.
    pub fn encode_batch(&self, data: &[F]) -> Vec<u8> {
        assert_eq!(data.len() % self.dim, 0, "data length must be a multiple of dim");
        let mut codes = vec![0; data.len()];
        for (code, vec) in codes.chunks_exact_mut(self.dim).zip(data.chunks_exact(self.dim)) {
            self.encode_into(vec, code);
        }
        codes
    }

    /// Reconstructs one vector from its `dim` codes.
    pub fn decode(&self, code: &[u8], out: &mut [F]) {
        assert_eq!(code.len(), self.dim, "code must have dim bytes");
        assert_eq!(out.len(), self.dim, "output must have dim values");
        for d in 0..self.dim {
            out[d] = reconstruct(code[d], self.min[d], self.step[d]);
        }
    }

    /// Asymmetric distances: `out[i] = ||query - decode(code_i)||^2`, decoding
    /// on the fly.
    pub fn distances(&self, query: &[F], codes: &[u8], out: &mut [F]) {
        let dim = self.dim;
        assert_eq!(query.len(), dim, "query must have dim values");
        assert_eq!(codes.len(), out.len() * dim, "codes must have dim bytes per output");
        let (query, min, step) = (&query[..dim], &self.min[..dim], &self.step[..dim]);
        // LLVM won't vectorize a loop that also accumulates an in-order float
        // sum, so square the differences into a buffer first and sum it after.
        let mut squares = [F::ZERO; BLOCK];
        for (slot, code) in out.iter_mut().zip(codes.chunks_exact(dim)) {
            let mut distance = F::ZERO;
            let mut start = 0;
            while start < dim {
                let len = BLOCK.min(dim - start);
                let code = &code[start..start + len];
                let query = &query[start..start + len];
                let min = &min[start..start + len];
                let step = &step[start..start + len];
                let squares = &mut squares[..len];
                for i in 0..len {
                    let diff = query[i] - reconstruct(code[i], min[i], step[i]);
                    squares[i] = diff * diff;
                }
                for &square in squares.iter() {
                    distance += square;
                }
                start += len;
            }
            *slot = distance;
        }
    }

    /// Top-`n` nearest encoded vectors by asymmetric distance, ascending.
    /// `codes` is a flat array of `num_codes * dim` bytes.
    pub fn search(&self, query: &[F], codes: &[u8], n: usize) -> Vec<(usize, F)> {
        let num_codes = codes.len() / self.dim;
        let mut dists = vec![F::ZERO; num_codes];
        self.distances(query, &codes[..num_codes * self.dim], &mut dists);
        top_n(&dists, n)
    }

    fn encode_into(&self, vec: &[F], code: &mut [u8]) {
        assert_eq!(vec.len(), self.dim, "vector must have dim values");
        for d in 0..self.dim {
            code[d] = if self.step[d] == F::ZERO {
                0 // constant dimension
            } else {
                // to_u8 clamps the bucket to [0, 255].
                ((vec[d] - self.min[d]) / self.step[d]).floor().to_u8()
            };
        }
    }
}

fn reconstruct<F: Float>(code: u8, min: F, step: F) -> F {
    min + (F::from_u8(code) + F::HALF) * step
}
