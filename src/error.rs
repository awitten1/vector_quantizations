use std::fmt;

/// Invalid training parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// `dim` is zero.
    InvalidDim,
    /// `dim` is zero, `m` is zero, or `m` does not divide `dim`.
    InvalidSubspaces,
    /// `k` is outside `[1, 256]`.
    InvalidK,
    /// The data length is not a multiple of `dim`.
    DataLengthMismatch,
    /// Fewer training vectors than centroids.
    TooFewVectors,
    /// No training vectors at all.
    NoTrainingVectors,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Error::InvalidDim => "dim must be positive",
            Error::InvalidSubspaces => "dim must be a positive multiple of m",
            Error::InvalidK => "k must be in [1, 256]",
            Error::DataLengthMismatch => "data length must be a multiple of dim",
            Error::TooFewVectors => "need at least k training vectors",
            Error::NoTrainingVectors => "need at least one training vector",
        })
    }
}

impl std::error::Error for Error {}
