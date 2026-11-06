//! Minimal walkthrough: product quantization (train, encode, decode, search),
//! then scalar quantization on the same data for comparison.

use vq::rng::{Mt19937_64, Normal};
use vq::{ProductQuantizer, ScalarQuantizer, l2_distance};

fn main() -> Result<(), vq::Error> {
    // 1. Data: n vectors of dimension dim, stored row-major in one flat array.
    //    Vector i lives at data[i * dim..(i + 1) * dim].
    let (n, dim) = (1000, 8);
    let mut rng = Mt19937_64::new(0);
    let mut normal = Normal::standard();
    let data: Vec<f32> = (0..n * dim).map(|_| normal.sample(&mut rng)).collect();
    let row_bytes = dim * size_of::<f32>();

    // 2. Train: split each vector into m = 4 pieces of dsub = 2 numbers,
    //    and learn k = 16 centroids per piece with k-means.
    let (m, k, iters) = (4, 16, 20);
    let pq = ProductQuantizer::train(&data, dim, m, k, iters)?;
    println!("trained: dim={} m={} k={} dsub={}", pq.dim(), pq.m(), pq.k(), pq.dsub());

    // 3. Encode one vector: each piece becomes the index of its nearest centroid.
    let v = &data[..dim]; // vector 0
    let code = pq.encode(v);
    print!("\nvector 0: ");
    for x in v {
        print!("{x:6.2} ");
    }
    print!("\ncode:     ");
    for c in &code {
        print!("{c} ");
    }
    println!("  ({m} bytes instead of {row_bytes})");

    // 4. Decode: look the centroids back up to get an approximation.
    let mut approx = vec![0.0; dim];
    pq.decode(&code, &mut approx);
    print!("decoded:  ");
    for x in &approx {
        print!("{x:6.2} ");
    }
    println!("\nsquared error: {:.3}", l2_distance(v, &approx));

    // 5. Search: encode the whole dataset, then find the vectors closest to a query
    //    using only the codes (asymmetric distance via a lookup table).
    let codes = pq.encode_batch(&data);
    let query: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
    let exact = |id: usize| l2_distance(&query, &data[id * dim..(id + 1) * dim]);

    println!("\ntop 5 for a random query (approx distance vs exact distance):");
    for (id, approx_dist) in pq.search(&query, &codes, 5) {
        println!("  id {id:4}  approx {approx_dist:.3}  exact {:.3}", exact(id));
    }

    // 6. Scalar quantization: each number independently becomes one byte
    //    (256 levels between that dimension's training min and max).
    let sq = ScalarQuantizer::train(&data, dim)?;
    let sq_code = sq.encode(v);
    let mut sq_approx = vec![0.0; dim];
    sq.decode(&sq_code, &mut sq_approx);
    print!("\nscalar quantization of vector 0:\ncode:     ");
    for c in &sq_code {
        print!("{c:3} ");
    }
    print!("  ({dim} bytes instead of {row_bytes})\ndecoded:  ");
    for x in &sq_approx {
        print!("{x:6.2} ");
    }
    println!(
        "\nsquared error: {:.5}  (PQ: {:.3})",
        l2_distance(v, &sq_approx),
        l2_distance(v, &approx)
    );

    let sq_codes = sq.encode_batch(&data);
    println!("\nSQ top 5 for the same query:");
    for (id, approx_dist) in sq.search(&query, &sq_codes, 5) {
        println!("  id {id:4}  approx {approx_dist:.3}  exact {:.3}", exact(id));
    }
    Ok(())
}
