#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerError<K> {
    NotFound(K),
    AlreadyBorrowedMutably(K),
    CapacityExceeded,
}

impl<K: core::fmt::Debug> core::fmt::Display for TrackerError<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "Key {:?} not tracked", key),
            Self::AlreadyBorrowedMutably(key) => {
                write!(f, "Key {:?} is already borrowed mutably", key)
            }
            Self::CapacityExceeded => write!(f, "Maximum tracker capacity reached"),
        }
    }
}

impl<K: core::fmt::Debug> std::error::Error for TrackerError<K> {}

/// Memory allocation via the C allocator (`malloc` / `realloc`) failed.
///
/// This is a stable equivalent of [`std::alloc::AllocError`] (nightly-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError;

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("C allocator failed (malloc/realloc returned null)")
    }
}

impl std::error::Error for AllocError {}
