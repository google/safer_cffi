// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::errors::TrackerError;
use crate::handle::Handle;
use crate::tracker::{ArcTrackerStorage, TrackedRef, TrackedRefMut, TrackerStorage};

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Mutex;

/// A tracker that manages the lifecycle of Rust objects across FFI boundaries.
///
/// The tracker stores Rust objects and identifies each of them by a generational handle
/// (`Handle<T>`) that can be passed to C code. It ensures that objects are only accessed when
/// valid, and prevents concurrent mutable access errors.
///
/// `S` is the `Box<T>` or `Arc<T>` storage type that the tracker holds (see [`TrackerStorage`]):
/// `Box<T>` by default, or `Arc<T>` to also allow shared borrows with [`OpaqueTracker::borrow`].
pub struct OpaqueTracker<T, S = Box<T>> {
    state: Mutex<OpaqueTrackerState<S>>,
    // `T` only appears in the keys. This marker keeps `S` in charge of the auto traits, so that
    // the tracker is `Sync` if `T: Send` for `Box<T>`, and if `T: Send + Sync` for `Arc<T>`.
    _key: PhantomData<fn() -> T>,
}

struct Entry<S> {
    value: Option<S>,
    generation: usize,
}

struct OpaqueTrackerState<S> {
    entries: Vec<Entry<S>>,
    free_indices: VecDeque<usize>,
}

impl<T, S: TrackerStorage<T>> OpaqueTracker<T, S> {
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(OpaqueTrackerState {
                entries: Vec::new(),
                free_indices: VecDeque::new(),
            }),
            _key: PhantomData,
        }
    }
}

impl<T, S: TrackerStorage<T>> Default for OpaqueTracker<T, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S: TrackerStorage<T>> OpaqueTracker<T, S> {
    /// Registers a new object with the tracker, taking ownership.
    ///
    /// Returns a `Handle` that can be passed to C.
    pub fn register(&self, val: S) -> Result<Handle<T>, TrackerError> {
        let mut state = self.state.lock().unwrap();
        let index = if let Some(idx) = state.free_indices.pop_front() {
            idx
        } else {
            let idx = state.entries.len();
            if idx > Handle::<T>::MAX_INDEX {
                std::hint::cold_path();
                drop(state);
                log::error!("Failed to register {}: CapacityExceeded", std::any::type_name::<T>());
                return Err(TrackerError::CapacityExceeded);
            }
            state.entries.push(Entry { value: None, generation: 1 });
            idx
        };

        let entry = &mut state.entries[index];
        entry.value = Some(val);

        let handle = Handle::new(index, entry.generation);
        Ok(handle)
    }

    /// Reclaims ownership of the object from the tracker.
    ///
    /// This removes the object from the tracker and returns the `Box`/`Arc` that was registered.
    /// Fails if the object is currently borrowed mutably.
    ///
    /// If the tracker holds `Arc<T>`, shared borrows that are still alive do not make this fail:
    /// the handle becomes invalid right away, and the object is dropped once the returned `Arc<T>`
    /// and the last [`TrackedRef`] are gone.
    pub fn reclaim(&self, key: impl Into<Handle<T>>) -> Result<S, TrackerError> {
        let key = key.into();
        let mut state = self.state.lock().unwrap();
        let index = key.index();

        let Some(entry) = state.entries.get_mut(index) else {
            std::hint::cold_path();
            drop(state);
            log::error!(
                "Failed to reclaim {}: NotFound for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::NotFound);
        };

        if entry.generation != key.generation() {
            std::hint::cold_path();
            drop(state);
            log::error!(
                "Failed to reclaim {}: Generation mismatch for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::NotFound);
        }

        let Some(val) = entry.value.take() else {
            std::hint::cold_path();
            drop(state);
            log::error!(
                "Failed to reclaim {}: AlreadyBorrowedMutably for key {:?}",
                std::any::type_name::<T>(),
                key
            );
            return Err(TrackerError::AlreadyBorrowedMutably);
        };

        if entry.generation < Handle::<T>::MAX_GENERATION {
            entry.generation += 1;
            state.free_indices.push_back(index);
        } else {
            std::hint::cold_path();
        }

        Ok(val)
    }

    /// Borrows the object, returning a guard that gives shared access.
    ///
    /// Only trackers that hold `Arc<T>` support shared borrows. Any number of [`TrackedRef`]
    /// guards for the same object can be alive at the same time, also on different threads. The
    /// borrow is released when the guard is dropped.
    ///
    /// Fails if the object is currently borrowed mutably.
    pub fn borrow(&self, key: impl Into<Handle<T>>) -> Result<TrackedRef<'_, T>, TrackerError>
    where
        S: ArcTrackerStorage<T>,
    {
        let key = key.into();
        let val = {
            let state = self.state.lock().unwrap();
            let index = key.index();

            let Some(entry) = state.entries.get(index) else {
                std::hint::cold_path();
                drop(state);
                log::error!(
                    "Failed to borrow {}: NotFound for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound);
            };

            if entry.generation != key.generation() {
                std::hint::cold_path();
                drop(state);
                log::error!(
                    "Failed to borrow {}: Generation mismatch for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound);
            }

            let Some(val) = entry.value.as_ref() else {
                std::hint::cold_path();
                drop(state);
                log::error!(
                    "Failed to borrow {}: AlreadyBorrowedMutably for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::AlreadyBorrowedMutably);
            };

            ArcTrackerStorage::share(val)
        };

        Ok(TrackedRef::new(val))
    }

    /// Borrows the object, returning a guard that gives mutable access.
    ///
    /// Returns a [`TrackedRefMut`] guard that implements `DerefMut`. When the guard is dropped,
    /// the object is automatically returned to the tracker.
    ///
    /// Fails if the object is currently borrowed. If the tracker holds `Arc<T>`, this includes any
    /// other `Arc<T>` or `Weak<T>` to the object, whether held by a [`TrackedRef`] or by other
    /// code.
    pub fn borrow_mut(
        &self,
        key: impl Into<Handle<T>>,
    ) -> Result<TrackedRefMut<'_, T, S>, TrackerError> {
        let key = key.into();
        let val = {
            let mut state_guard = self.state.lock().unwrap();
            let state = &mut *state_guard;
            let index = key.index();

            let Some(entry) = state.entries.get_mut(index) else {
                std::hint::cold_path();
                drop(state_guard);
                log::error!(
                    "Failed to borrow_mut {}: NotFound for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound);
            };

            if entry.generation != key.generation() {
                std::hint::cold_path();
                drop(state_guard);
                log::error!(
                    "Failed to borrow_mut {}: Generation mismatch for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::NotFound);
            }
            let Some(mut val) = entry.value.take() else {
                std::hint::cold_path();
                drop(state_guard);
                log::error!(
                    "Failed to borrow_mut {}: AlreadyBorrowedMutably for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::AlreadyBorrowedMutably);
            };

            // A `Box<T>` is never shared. An `Arc<T>` is shared while `TrackedRef` guards or other
            // `Arc`/`Weak` pointers to it are alive.
            if TrackerStorage::get_mut(&mut val).is_none() {
                entry.value = Some(val);
                std::hint::cold_path();
                drop(state_guard);
                log::error!(
                    "Failed to borrow_mut {}: AlreadyBorrowedImmutably for key {:?}",
                    std::any::type_name::<T>(),
                    key
                );
                return Err(TrackerError::AlreadyBorrowedImmutably);
            }

            val
        };

        Ok(TrackedRefMut::new(self, key, val))
    }

    /// Returns the borrowed object back to the tracker.
    ///
    /// Only the `Drop` implementation of the [`TrackedRefMut`] guard calls this.
    pub(crate) fn return_mut(&self, key: Handle<T>, val: S) {
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
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

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

        assert_eq!(tracker.borrow_mut(handle).err().unwrap(), TrackerError::NotFound);
    }

    #[gtest]
    fn test_null_handle() -> Result<()> {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let null: Handle<usize> = Handle::null();

        verify_that!(tracker.borrow_mut(null).err(), some(eq(TrackerError::NotFound)))?;
        verify_that!(tracker.reclaim(null).err(), some(eq(TrackerError::NotFound)))
    }

    #[gtest]
    fn test_reused_slot_gets_new_generation() -> Result<()> {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let stale = tracker.register(Box::new(1))?;
        tracker.reclaim(stale)?;

        // The freed slot is reused with a higher generation, so the old handle stays invalid.
        let fresh = tracker.register(Box::new(2))?;

        verify_that!(fresh.index(), eq(stale.index()))?;
        verify_that!(fresh.generation(), gt(stale.generation()))?;
        verify_that!(tracker.borrow_mut(stale).err(), some(eq(TrackerError::NotFound)))?;
        verify_that!(tracker.reclaim(stale).err(), some(eq(TrackerError::NotFound)))?;
        let value = *tracker.borrow_mut(fresh)?;
        verify_that!(value, eq(2))
    }

    #[gtest]
    fn test_double_borrow() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let handle = tracker.register(Box::new(1)).unwrap();

        let _tracked = tracker.borrow_mut(handle).unwrap();
        assert_eq!(tracker.borrow_mut(handle).err().unwrap(), TrackerError::AlreadyBorrowedMutably);
    }

    #[gtest]
    fn test_reclaim_while_borrowed() {
        let tracker: OpaqueTracker<usize> = OpaqueTracker::new();
        let handle = tracker.register(Box::new(1)).unwrap();

        let _tracked = tracker.borrow_mut(handle).unwrap();
        assert_eq!(tracker.reclaim(handle).unwrap_err(), TrackerError::AlreadyBorrowedMutably);
    }

    #[gtest]
    fn test_box_tracker_is_sync_for_objects_that_are_only_send() {
        fn assert_sync<S: Sync>() {}

        // `Cell` is `Send`, but not `Sync`. Exclusive borrows never share an object between
        // threads, so a tracker that holds `Box<T>` must remain usable as a `static`.
        assert_sync::<OpaqueTracker<Cell<usize>>>();
    }

    type SharedTracker = OpaqueTracker<usize, Arc<usize>>;

    #[gtest]
    fn test_shared_borrows_coexist() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(42))?;

        let first = tracker.borrow(handle)?;
        let second = tracker.borrow(handle)?;

        verify_that!(*first, eq(42))?;
        verify_that!(*second, eq(42))
    }

    #[gtest]
    fn test_borrow_mut_after_shared_borrows_are_released() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(42))?;
        drop(tracker.borrow(handle)?);

        *tracker.borrow_mut(handle)? = 99;

        verify_that!(*tracker.borrow(handle)?, eq(99))?;
        verify_that!(*tracker.reclaim(handle)?, eq(99))
    }

    #[gtest]
    fn test_borrow_mut_while_shared_borrowed() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(1))?;
        let _shared = tracker.borrow(handle)?;

        let result = tracker.borrow_mut(handle);

        verify_that!(result.err(), some(eq(TrackerError::AlreadyBorrowedImmutably)))
    }

    #[gtest]
    fn test_borrow_mut_while_arc_is_shared_outside_of_tracker() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let object = Arc::new(1);
        let handle = tracker.register(Arc::clone(&object))?;

        let result = tracker.borrow_mut(handle);

        verify_that!(result.err(), some(eq(TrackerError::AlreadyBorrowedImmutably)))?;
        drop(object);
        verify_that!(tracker.borrow_mut(handle).err(), none())
    }

    #[gtest]
    fn test_borrow_mut_while_weak_exists() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let object = Arc::new(1);
        let weak = Arc::downgrade(&object);
        let handle = tracker.register(object)?;

        let result = tracker.borrow_mut(handle);

        verify_that!(result.err(), some(eq(TrackerError::AlreadyBorrowedImmutably)))?;
        drop(weak);
        verify_that!(tracker.borrow_mut(handle).err(), none())
    }

    #[gtest]
    fn test_borrow_while_mutably_borrowed() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(1))?;
        let _mutable = tracker.borrow_mut(handle)?;

        let result = tracker.borrow(handle);

        verify_that!(result.err(), some(eq(TrackerError::AlreadyBorrowedMutably)))
    }

    #[gtest]
    fn test_borrow_stale_handle() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(1))?;
        tracker.reclaim(handle)?;

        let result = tracker.borrow(handle);

        verify_that!(result.err(), some(eq(TrackerError::NotFound)))
    }

    #[gtest]
    fn test_reclaim_while_shared_borrowed_keeps_object_alive() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(1))?;
        let shared = tracker.borrow(handle)?;

        let reclaimed = tracker.reclaim(handle)?;

        // The handle is invalid from now on, but the shared borrow can still be used.
        verify_that!(tracker.borrow(handle).err(), some(eq(TrackerError::NotFound)))?;
        verify_that!(*shared, eq(1))?;
        verify_that!(Arc::strong_count(&reclaimed), eq(2))?;
        drop(shared);
        verify_that!(Arc::strong_count(&reclaimed), eq(1))
    }

    #[gtest]
    fn test_reclaim_while_mutably_borrowed_with_arc() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(1))?;
        let _mutable = tracker.borrow_mut(handle)?;

        let result = tracker.reclaim(handle);

        verify_that!(result.err(), some(eq(TrackerError::AlreadyBorrowedMutably)))
    }

    const READER_THREADS: usize = 2;
    const WRITE_ATTEMPTS: usize = 400;

    /// What happened to a counter that several threads borrowed at the same time.
    struct Outcome {
        /// The counter, which was reclaimed while the threads were running.
        reclaimed: Arc<usize>,
        /// How often the writer thread incremented the counter.
        writes: usize,
        /// The first thing a thread saw that must never happen, if any.
        violation: Option<String>,
    }

    /// Lets `READER_THREADS` threads read the counter behind `handle` while another thread
    /// increments it, and reclaims the counter while all of them are running.
    fn borrow_from_several_threads(
        tracker: &SharedTracker,
        handle: Handle<usize>,
    ) -> Result<Outcome> {
        let start = Barrier::new(READER_THREADS + 2);
        let halfway = Barrier::new(2);
        let reclaiming = AtomicBool::new(false);
        let done = AtomicBool::new(false);
        let writes = AtomicUsize::new(0);
        let first_violation = Mutex::new(None);
        let report = |violation: String| {
            first_violation
                .lock()
                .expect("no thread panics while it holds this lock")
                .get_or_insert(violation);
        };

        let reader = || {
            start.wait();
            let mut last_seen = 0;
            while !done.load(Ordering::SeqCst) {
                match tracker.borrow(handle) {
                    Ok(counter) => {
                        if *counter < last_seen {
                            report(format!("a reader saw {} after {last_seen}", *counter));
                        }
                        last_seen = *counter;
                        // Keep the shared borrow for a moment, so that the writer runs into it.
                        thread::yield_now();
                    }
                    // The writer has the counter.
                    Err(TrackerError::AlreadyBorrowedMutably) => {}
                    Err(TrackerError::NotFound) if reclaiming.load(Ordering::SeqCst) => break,
                    Err(err) => report(format!("a reader got {err:?}")),
                }
                // Leave a gap between two borrows, so that the writer gets its turn.
                thread::yield_now();
            }
        };
        let writer = || {
            start.wait();
            for attempt in 0..WRITE_ATTEMPTS {
                if attempt == WRITE_ATTEMPTS / 2 {
                    halfway.wait();
                }
                match tracker.borrow_mut(handle) {
                    Ok(mut counter) => {
                        *counter += 1;
                        writes.fetch_add(1, Ordering::SeqCst);
                        // Keep the exclusive borrow for a moment, so that the readers run into it.
                        thread::yield_now();
                    }
                    // A reader has the counter.
                    Err(TrackerError::AlreadyBorrowedImmutably) => {}
                    Err(TrackerError::NotFound) if reclaiming.load(Ordering::SeqCst) => break,
                    Err(err) => report(format!("the writer got {err:?}")),
                }
                // Leave a gap between two borrows, so that the readers get their turn.
                thread::yield_now();
            }
        };

        let reclaimed = thread::scope(|scope| {
            for _ in 0..READER_THREADS {
                scope.spawn(reader);
            }
            scope.spawn(writer);

            start.wait();
            halfway.wait();
            reclaiming.store(true, Ordering::SeqCst);
            let reclaimed = loop {
                match tracker.reclaim(handle) {
                    // The writer has the counter.
                    Err(TrackerError::AlreadyBorrowedMutably) => thread::yield_now(),
                    result => break result,
                }
            };
            done.store(true, Ordering::SeqCst);
            reclaimed
        });

        Ok(Outcome {
            reclaimed: reclaimed?,
            writes: writes.into_inner(),
            violation: first_violation
                .into_inner()
                .expect("no thread panicked while it held this lock"),
        })
    }

    #[gtest]
    fn test_borrows_from_several_threads() -> Result<()> {
        let tracker: SharedTracker = OpaqueTracker::new();
        let handle = tracker.register(Arc::new(0))?;

        let outcome = borrow_from_several_threads(&tracker, handle)?;

        verify_that!(outcome.violation, none())?;
        // No increment was lost.
        verify_that!(*outcome.reclaimed, eq(outcome.writes))?;
        // Every shared borrow was released.
        verify_that!(Arc::strong_count(&outcome.reclaimed), eq(1))
    }
}
