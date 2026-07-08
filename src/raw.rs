use crate::errors::TrackerError;
use crate::tracker::Tracker;

use std::collections::BTreeMap;
use std::sync::Mutex;

/// A tracker that manages objects using their raw memory addresses (`*mut T`) as keys.
///
/// Caveats:
///
/// - Cannot be used if the API allows for objects to be created on the C side.
/// - Does not prevent ABA problems.
/// - Slightly slower than the opaque tracker.
pub struct RawTracker<T> {
    // Switching this to a HashMap yields a 2x speedup for the `raw_many` benchmark,
    // but a 77% slowdown for the `raw_single` benchmark.
    state: Mutex<BTreeMap<usize, Entry<T>>>,
}

struct Entry<T> {
    value: Option<Box<T>>,
}

impl<T> RawTracker<T> {
    pub const fn new() -> Self {
        Self { state: Mutex::new(BTreeMap::new()) }
    }
}

impl<T> Default for RawTracker<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Tracker<T> for RawTracker<T> {
    type Key = *mut T;

    fn register(&self, b: Box<T>) -> Result<Self::Key, TrackerError<Self::Key>> {
        let mut state = self.state.lock().unwrap();
        let ptr = &*b as *const T as *mut T;
        let addr = ptr as usize;

        if state.contains_key(&addr) {
            std::hint::cold_path();
            log::error!("Failed to register {}: CapacityExceeded", std::any::type_name::<T>());
            return Err(TrackerError::CapacityExceeded);
        }

        state.insert(addr, Entry { value: Some(b) });
        Ok(ptr)
    }

    fn reclaim(&self, key: impl Into<Self::Key>) -> Result<Box<T>, TrackerError<Self::Key>> {
        let key = key.into();
        let mut state = self.state.lock().unwrap();
        let addr = key as usize;

        let Some(entry) = state.get_mut(&addr) else {
            std::hint::cold_path();
            log::error!(
                "Failed to reclaim {}: NotFound for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::NotFound(key));
        };

        let Some(val) = entry.value.take() else {
            std::hint::cold_path();
            log::error!(
                "Failed to reclaim {}: AlreadyBorrowedMutably for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::AlreadyBorrowedMutably(key));
        };

        state.remove(&addr);
        Ok(val)
    }

    fn borrow_mut(
        &self,
        key: impl Into<Self::Key>,
    ) -> Result<crate::Tracked<'_, T, Self>, TrackerError<Self::Key>> {
        let key = key.into();
        let val = {
            let mut state = self.state.lock().unwrap();
            let addr = key as usize;

            let Some(entry) = state.get_mut(&addr) else {
                std::hint::cold_path();
                log::error!(
                    "Failed to borrow_mut {}: NotFound for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound(key));
            };

            let Some(val) = entry.value.take() else {
                std::hint::cold_path();
                log::error!(
                    "Failed to borrow_mut {}: AlreadyBorrowedMutably for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::AlreadyBorrowedMutably(key));
            };

            val
        };

        Ok(crate::Tracked::new(self, key, val))
    }

    fn return_mut(&self, key: Self::Key, val: Box<T>) {
        let mut state = self.state.lock().unwrap();
        let addr = key as usize;
        let entry = state.get_mut(&addr).expect("corrupted tracker state: key not found");
        entry.value = Some(val);
    }
}

#[cfg(test)]
mod tests {
        use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_basic() {
        let tracker: RawTracker<usize> = RawTracker::new();
        let val = 999;
        let key = tracker.register(Box::new(val)).unwrap();

        {
            let mut tracked = tracker.borrow_mut(key).unwrap();
            assert_eq!(*tracked, 999);
            *tracked = 1111;
        }

        let res = tracker.reclaim(key).unwrap();
        assert_eq!(*res, 1111);
    }

    #[gtest]
    fn test_double_borrow() {
        let tracker: RawTracker<usize> = RawTracker::new();
        let key = tracker.register(Box::new(1)).unwrap();

        let _tracked = tracker.borrow_mut(key).unwrap();
        assert_eq!(
            tracker.borrow_mut(key).err().unwrap(),
            TrackerError::AlreadyBorrowedMutably(key)
        );
    }

    #[gtest]
    fn test_reclaim_while_borrowed() {
        let tracker: RawTracker<usize> = RawTracker::new();
        let key = tracker.register(Box::new(1)).unwrap();

        let _tracked = tracker.borrow_mut(key).unwrap();
        assert_eq!(tracker.reclaim(key).unwrap_err(), TrackerError::AlreadyBorrowedMutably(key));
    }
}
