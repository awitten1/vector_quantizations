use crate::common::{Float, l2_distance, top_n};
use crate::error::Error;
use crate::rng::{self, Mt19937_64};

/// Training seed used by [`ProductQuantizer::train`].
pub const DEFAULT_SEED: u64 = 42;

/// k-means++ seeding over the contiguous `dim`-dimensional points in `data`.
/// Writes `k` centroids to `centroids` (`k * dim` values). Each new centroid is
/// a training point sampled with probability proportional to its squared
/// distance to the nearest centroid chosen so far. Panics if `data` holds no
/// complete point or `centroids` has the wrong length.
pub fn kmeans_plus_plus<F: Float>(
    data: &[F],
    dim: usize,
    k: usize,
    centroids: &mut [F],
    rng: &mut Mt19937_64,
) {
    assert!(dim > 0, "dim must be positive");
    let n = data.len() / dim;
    assert!(n >= 1, "need at least one point");
    assert_eq!(centroids.len(), k * dim, "centroids must hold k * dim values");

    let mut min_dist = vec![f64::MAX; n];
    let mut chosen = rng::uniform_below(rng, n as u64) as usize;
    for c in 0..k {
        let centroid = &mut centroids[c * dim..(c + 1) * dim];
        centroid.copy_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
        if c + 1 == k {
            break;
        }
        for (min, x) in min_dist.iter_mut().zip(data.chunks_exact(dim)) {
            let distance = l2_distance(x, centroid).to_f64();
            if distance < *min {
                *min = distance;
            }
        }
        let total: f64 = min_dist.iter().fold(0.0, |sum, d| sum + d);
        chosen = if total > 0.0 {
            rng::discrete(&min_dist, rng)
        } else {
            // Every point already coincides with a centroid.
            rng::uniform_below(rng, n as u64) as usize
        };
    }
}

/// Product quantizer: splits each `dim`-dimensional vector into `m` subvectors
/// of `dsub = dim / m` and learns a `k`-centroid codebook per subspace with
/// k-means. Each vector is then stored as `m` one-byte codes.
#[derive(Clone, Debug)]
pub struct ProductQuantizer<F> {
    dim: usize,
    m: usize,
    k: usize,
    dsub: usize,
    codebooks: Vec<F>, // [m][k][dsub]
}

impl<F: Float> ProductQuantizer<F> {
    /// Trains with [`DEFAULT_SEED`]. See [`Self::train_with_seed`].
    pub fn train(data: &[F], dim: usize, m: usize, k: usize, iters: usize) -> Result<Self, Error> {
        Self::train_with_seed(data, dim, m, k, iters, DEFAULT_SEED)
    }

    /// Trains on the contiguous `dim`-dimensional vectors in `data`.
    ///
    /// `m` is the number of subspaces (it must divide `dim`), `k` the number of
    /// centroids per subspace (at most 256), and `iters` the maximum number of
    /// k-means iterations.
    pub fn train_with_seed(
        data: &[F],
        dim: usize,
        m: usize,
        k: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, Error> {
        if dim == 0 || m == 0 || dim % m != 0 {
            return Err(Error::InvalidSubspaces);
        }
        if k == 0 || k > 256 {
            return Err(Error::InvalidK);
        }
        if data.len() % dim != 0 {
            return Err(Error::DataLengthMismatch);
        }
        let n = data.len() / dim;
        if n < k {
            return Err(Error::TooFewVectors);
        }
        let dsub = dim / m;
        let mut codebooks = vec![F::ZERO; m * k * dsub];

        // Gather each subspace into a contiguous n * dsub buffer so k-means
        // streams through memory instead of striding across full vectors.
        let mut sub_data = vec![F::ZERO; n * dsub];
        for (sub, centroids) in codebooks.chunks_exact_mut(k * dsub).enumerate() {
            for (dst, vec) in sub_data.chunks_exact_mut(dsub).zip(data.chunks_exact(dim)) {
                dst.copy_from_slice(&vec[sub * dsub..(sub + 1) * dsub]);
            }
            kmeans(&sub_data, dsub, k, iters, centroids, seed.wrapping_add(sub as u64));
        }
        Ok(Self { dim, m, k, dsub, codebooks })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Centroid `c` of subspace `sub`. Panics if either is out of range.
    pub fn centroid(&self, sub: usize, c: usize) -> &[F] {
        assert!(sub < self.m && c < self.k, "centroid out of range");
        &self.codebooks[(sub * self.k + c) * self.dsub..][..self.dsub]
    }

    /// Encodes one vector into `m` codes.
    pub fn encode(&self, vec: &[F]) -> Vec<u8> {
        let mut code = vec![0; self.m];
        self.encode_into(vec, &mut code);
        code
    }

    /// Encodes contiguous vectors into a flat array of `n * m` codes. Panics if
    /// `data` is not a whole number of vectors.
    pub fn encode_batch(&self, data: &[F]) -> Vec<u8> {
        assert_eq!(data.len() % self.dim, 0, "data length must be a multiple of dim");
        let mut codes = vec![0; data.len() / self.dim * self.m];
        for (code, vec) in codes.chunks_exact_mut(self.m).zip(data.chunks_exact(self.dim)) {
            self.encode_into(vec, code);
        }
        codes
    }

    /// Reconstructs one vector from its `m` codes.
    pub fn decode(&self, code: &[u8], out: &mut [F]) {
        assert_eq!(code.len(), self.m, "code must have m bytes");
        assert_eq!(out.len(), self.dim, "output must have dim values");
        for (sub, (dst, &c)) in out.chunks_exact_mut(self.dsub).zip(code).enumerate() {
            dst.copy_from_slice(self.centroid(sub, usize::from(c)));
        }
    }

    /// ADC lookup table: `table[sub * k + c] = ||query_sub - centroid(sub, c)||^2`.
    pub fn distance_table(&self, query: &[F]) -> Vec<F> {
        assert_eq!(query.len(), self.dim, "query must have dim values");
        let mut table = Vec::with_capacity(self.m * self.k);
        for (query_sub, codebook) in query
            .chunks_exact(self.dsub)
            .zip(self.codebooks.chunks_exact(self.k * self.dsub))
        {
            for centroid in codebook.chunks_exact(self.dsub) {
                table.push(l2_distance(query_sub, centroid));
            }
        }
        table
    }

    /// Asymmetric distances from a query to `out.len()` encoded vectors, using
    /// the query's [`distance_table`](Self::distance_table):
    /// `out[i]` is the sum over `sub` of `table[sub * k + code_i[sub]]`.
    pub fn adc_distances(&self, table: &[F], codes: &[u8], out: &mut [F]) {
        assert_eq!(table.len(), self.m * self.k, "table must have m * k values");
        assert_eq!(codes.len(), out.len() * self.m, "codes must have m bytes per output");
        for (slot, code) in out.iter_mut().zip(codes.chunks_exact(self.m)) {
            let mut distance = F::ZERO;
            for (sub, &c) in code.iter().enumerate() {
                distance += table[sub * self.k + usize::from(c)];
            }
            *slot = distance;
        }
    }

    /// Top-`n` nearest encoded vectors by asymmetric distance, ascending.
    /// `codes` is a flat array of `num_codes * m` bytes.
    pub fn search(&self, query: &[F], codes: &[u8], n: usize) -> Vec<(usize, F)> {
        let num_codes = codes.len() / self.m;
        let table = self.distance_table(query);
        let mut distances = vec![F::ZERO; num_codes];
        self.adc_distances(&table, &codes[..num_codes * self.m], &mut distances);
        top_n(&distances, n)
    }

    fn encode_into(&self, vec: &[F], code: &mut [u8]) {
        assert_eq!(vec.len(), self.dim, "vector must have dim values");
        for ((c, subvec), codebook) in code
            .iter_mut()
            .zip(vec.chunks_exact(self.dsub))
            .zip(self.codebooks.chunks_exact(self.k * self.dsub))
        {
            *c = nearest(subvec, codebook, self.dsub) as u8;
        }
    }
}

/// Index of the centroid in `codebook` (contiguous `dsub`-dimensional
/// centroids) closest to `subvec`.
fn nearest<F: Float>(subvec: &[F], codebook: &[F], dsub: usize) -> usize {
    let mut closest = 0;
    let mut closest_distance = F::MAX;
    for (c, centroid) in codebook.chunks_exact(dsub).enumerate() {
        let distance = l2_distance(subvec, centroid);
        if distance < closest_distance {
            closest_distance = distance;
            closest = c;
        }
    }
    closest
}

/// Lloyd's k-means on the contiguous `dsub`-dimensional points in `data`,
/// writing `k` centroids to `centroids`.
fn kmeans<F: Float>(
    data: &[F],
    dsub: usize,
    k: usize,
    iters: usize,
    centroids: &mut [F],
    seed: u64,
) {
    const UNASSIGNED: usize = usize::MAX;

    let n = data.len() / dsub;
    let mut rng = Mt19937_64::new(seed);

    kmeans_plus_plus(data, dsub, k, centroids, &mut rng);

    let mut assign = vec![UNASSIGNED; n];
    let mut counts = vec![0usize; k];
    for _ in 0..iters {
        // Assignment step.
        let mut changed = false;
        for (slot, x) in assign.iter_mut().zip(data.chunks_exact(dsub)) {
            let closest = nearest(x, centroids, dsub);
            if closest != *slot {
                *slot = closest;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step: each centroid becomes the mean of its members.
        centroids.fill(F::ZERO);
        counts.fill(0);
        for (&cluster, x) in assign.iter().zip(data.chunks_exact(dsub)) {
            let centroid = &mut centroids[cluster * dsub..(cluster + 1) * dsub];
            for (sum, &value) in centroid.iter_mut().zip(x) {
                *sum += value;
            }
            counts[cluster] += 1;
        }
        for (centroid, &count) in centroids.chunks_exact_mut(dsub).zip(&counts) {
            if count == 0 {
                // Empty cluster: re-seed from a random training point.
                let point = rng::uniform_below(&mut rng, n as u64) as usize;
                centroid.copy_from_slice(&data[point * dsub..(point + 1) * dsub]);
                continue;
            }
            let count = F::from_usize(count);
            for value in centroid {
                *value /= count;
            }
        }
    }
}
