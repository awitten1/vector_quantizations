#[test]
fn version_is_not_empty() {
    assert!(!vq::version().is_empty());
}

#[test]
fn top_n_is_ascending_and_breaks_ties_by_index() {
    let distances = [3.0f32, 1.0, f32::NAN, 1.0, 0.5];
    assert_eq!(vq::top_n(&distances, 3), [(4, 0.5), (1, 1.0), (3, 1.0)]);
    assert!(vq::top_n(&distances, 0).is_empty());
    let all = vq::top_n(&distances, 99);
    assert_eq!(all.len(), 5);
    assert_eq!(all[3], (0, 3.0));
    assert_eq!(all[4].0, 2);
}
