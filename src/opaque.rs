use crate::errors::TrackerError;
use crate::handle::Handle;
use crate::tracker::Tracker;

use std::collections::VecDeque;
use std::sync::Mutex;

/// A tracker that manages objects using generational handles (`Handle<T>`).
pub struct OpaqueTracker<T> {
    state: Mutex<OpaqueTrackerState<T>>,
}

struct Entry<T> {
    value: Option<Box<T>>,
    generation: usize,
}

struct OpaqueTrackerState<T> {
    entries: Vec<Entry<T>>,
    free_indices: VecDeque<usize>,
}

impl<T> OpaqueTracker<T> {
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(OpaqueTrackerState {
                entries: Vec::new(),
                free_indices: VecDeque::new(),
            }),
        }
    }
}

impl<T> Default for OpaqueTracker<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Tracker<T> for OpaqueTracker<T> {
    type Key = Handle<T>;

    fn register(&self, b: Box<T>) -> Result<Self::Key, TrackerError<Self::Key>> {
        let mut state = self.state.lock().unwrap();
        let index = if let Some(idx) = state.free_indices.pop_front() {
            idx
        } else {
            let idx = state.entries.len();
            if idx > Handle::<T>::MAX_INDEX {
                std::hint::cold_path();
                log::error!("Failed to register {}: CapacityExceeded", std::any::type_name::<T>());
                return Err(TrackerError::CapacityExceeded);
            }
            state.entries.push(Entry { value: None, generation: 1 });
            idx
        };

        let entry = &mut state.entries[index];
        entry.value = Some(b);

        let handle = Handle::new(index, entry.generation);
        Ok(handle)
    }

    fn reclaim(&self, key: impl Into<Self::Key>) -> Result<Box<T>, TrackerError<Self::Key>> {
        let key = key.into();
        let mut state = self.state.lock().unwrap();
        let index = key.index();

        let Some(entry) = state.entries.get_mut(index) else {
            std::hint::cold_path();
            log::error!(
                "Failed to reclaim {}: NotFound for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::NotFound(key));
        };

        if entry.generation != key.generation() {
            std::hint::cold_path();
            log::error!(
                "Failed to reclaim {}: Generation mismatch for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::NotFound(key));
        }

        let Some(val) = entry.value.take() else {
            std::hint::cold_path();
            log::error!(
                "Failed to reclaim {}: AlreadyBorrowedMutably for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::AlreadyBorrowedMutably(key));
        };

        if entry.generation < Handle::<T>::MAX_GENERATION {
            entry.generation += 1;
            state.free_indices.push_back(index);
        } else {
            std::hint::cold_path();
        }

        Ok(val)
    }

    fn borrow_mut(
        &self,
        key: impl Into<Self::Key>,
    ) -> Result<crate::Tracked<'_, T, Self>, TrackerError<Self::Key>> {
        let key = key.into();
        let val = {
            let mut state_guard = self.state.lock().unwrap();
            let state = &mut *state_guard;
            let index = key.index();

            let Some(entry) = state.entries.get_mut(index) else {
                std::hint::cold_path();
                log::error!(
                    "Failed to borrow_mut {}: NotFound for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound(key));
            };

            if entry.generation != key.generation() {
                std::hint::cold_path();
                log::error!(
                    "Failed to borrow_mut {}: Generation mismatch for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound(key));
            }
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
        let mut state_guard = self.state.lock().unwrap();
        let state = &mut *state_guard;
        let index = key.index();

        let entry =
            state.entries.get_mut(index).expect("corrupted tracker state: index out of bounds");
        debug_assert_eq!(
            entry.generation,
            key.generation(),
            "corrupted tracker state: generation mismatch"
        );

        entry.value = Some(val);
    }
}

#[cfg(test)]
mod tests {
        use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_basic() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let val = 999;
        let handle = tracker.register(Box::new(val)).unwrap();

        {
            let mut tracked = tracker.borrow_mut(handle).unwrap();
            assert_eq!(*tracked, 999);
            *tracked = 1111;
        }

        let res = tracker.reclaim(handle).unwrap();
        assert_eq!(*res, 1111);
    }

    #[gtest]
    fn test_basic_impl_into() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let val = 999;
        let handle = tracker.register(Box::new(val)).unwrap();
        let ptr = handle.into_opaque_ptr();

        {
            // Verify we can pass *mut T directly via Into
            let mut tracked = tracker.borrow_mut(ptr).unwrap();
            assert_eq!(*tracked, 999);
            *tracked = 1111;
        }

        // Verify we can pass *mut T directly via Into
        let res = tracker.reclaim(ptr).unwrap();
        assert_eq!(*res, 1111);
    }

    #[gtest]
    fn test_stale_handle() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let handle = tracker.register(Box::new(1)).unwrap();
        let _ = tracker.reclaim(handle).unwrap();

        assert_eq!(tracker.borrow_mut(handle).err().unwrap(), TrackerError::NotFound(handle));
    }

    #[gtest]
    fn test_double_borrow() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let handle = tracker.register(Box::new(1)).unwrap();

        let _tracked = tracker.borrow_mut(handle).unwrap();
        assert_eq!(
            tracker.borrow_mut(handle).err().unwrap(),
            TrackerError::AlreadyBorrowedMutably(handle)
        );
    }

    #[gtest]
    fn test_reclaim_while_borrowed() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let handle = tracker.register(Box::new(1)).unwrap();

        let _tracked = tracker.borrow_mut(handle).unwrap();
        assert_eq!(
            tracker.reclaim(handle).unwrap_err(),
            TrackerError::AlreadyBorrowedMutably(handle)
        );
    }
}
