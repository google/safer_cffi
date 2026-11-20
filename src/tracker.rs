// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::handle::Handle;
use crate::opaque::OpaqueTracker;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

mod sealed {
    use std::sync::Arc;

    pub trait Sealed {}
    impl<T> Sealed for Box<T> {}
    impl<T> Sealed for Arc<T> {}
}

/// Storage for an object held by an [`OpaqueTracker`].
///
/// The storage type decides how tracked objects can be borrowed:
///
/// - `Box<T>` (the default): the tracker is the only owner of the object, which can only be
///   borrowed exclusively with [`OpaqueTracker::borrow_mut`].
/// - `Arc<T>`: the object can additionally be borrowed shared with [`OpaqueTracker::borrow`].
///
/// This trait is sealed and cannot be implemented for other types.
pub trait TrackerStorage<T>: Deref<Target = T> + sealed::Sealed + Sized {
    /// Returns exclusive access to the object, or `None` while it is shared.
    #[doc(hidden)]
    fn get_mut(this: &mut Self) -> Option<&mut T>;
}

impl<T> TrackerStorage<T> for Box<T> {
    #[inline]
    fn get_mut(this: &mut Self) -> Option<&mut T> {
        Some(this)
    }
}

impl<T> TrackerStorage<T> for Arc<T> {
    #[inline]
    fn get_mut(this: &mut Self) -> Option<&mut T> {
        Arc::get_mut(this)
    }
}

/// Marker trait for [`TrackerStorage`] backed by `Arc<T>`, enabling shared borrows via
/// [`OpaqueTracker::borrow`].
///
/// This trait is sealed and cannot be implemented for other types.
// This really should be an `impl<T> OpaqueTracker<T, Arc<T>>` block, but to get good error
// messages when calling `.borrow()` on a `Box`-backed tracker, it needs to be a trait with
// `#[diagnostic::on_unimplemented]` (otherwise `rustc` emits E0599 and suggests importing
// `std::borrow::Borrow` instead of switching to `OpaqueTracker<T, Arc<T>>`).
#[diagnostic::on_unimplemented(
    message = "shared borrows need a tracker that holds `Arc<{T}>`, not `{Self}`",
    note = "declare the tracker as `OpaqueTracker<{T}, Arc<{T}>>`"
)]
pub trait ArcTrackerStorage<T>: TrackerStorage<T> {
    /// Returns another `Arc<T>` referencing the same object.
    #[doc(hidden)]
    fn share(this: &Self) -> Arc<T>;
}

impl<T> ArcTrackerStorage<T> for Arc<T> {
    #[inline]
    fn share(this: &Self) -> Arc<T> {
        Arc::clone(this)
    }
}

/// A guard that provides shared access to a tracked object.
///
/// The shared borrow is released when this guard is dropped.
pub struct TrackedRef<'tracker, T> {
    val: Arc<T>,
    // The guard keeps the object alive by itself. It borrows the tracker nevertheless, like
    // `TrackedRefMut`, so that a future guard can report back to its tracker when it is dropped.
    _tracker: PhantomData<&'tracker ()>,
}

impl<T> TrackedRef<'_, T> {
    /// Creates a guard that borrows the object held by `val`.
    pub(crate) fn new(val: Arc<T>) -> Self {
        Self { val, _tracker: PhantomData }
    }
}

impl<T> Deref for TrackedRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

/// A guard that provides mutable access to a tracked object.
///
/// The object is returned to the tracker when this guard is dropped.
pub struct TrackedRefMut<'tracker, T, S: TrackerStorage<T> = Box<T>> {
    tracker: &'tracker OpaqueTracker<T, S>,
    key: Handle<T>,
    val: Option<S>,
}

impl<'tracker, T, S: TrackerStorage<T>> TrackedRefMut<'tracker, T, S> {
    pub(crate) fn new(tracker: &'tracker OpaqueTracker<T, S>, key: Handle<T>, val: S) -> Self {
        Self { tracker, key, val: Some(val) }
    }
}

impl<T, S: TrackerStorage<T>> Deref for TrackedRefMut<'_, T, S> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.val.as_deref().expect("corrupted TrackedRefMut state: value is None")
    }
}

impl<T, S: TrackerStorage<T>> DerefMut for TrackedRefMut<'_, T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let val = self.val.as_mut().expect("corrupted TrackedRefMut state: value is None");
        // The tracker only hands out values that are not shared, and nothing can share the object
        // while it is held by this guard.
        TrackerStorage::get_mut(val).expect("corrupted TrackedRefMut state: value is shared")
    }
}

impl<T, S: TrackerStorage<T>> Drop for TrackedRefMut<'_, T, S> {
    fn drop(&mut self) {
        let v = self.val.take().expect("corrupted TrackedRefMut state: value is None");
        self.tracker.return_mut(self.key, v);
    }
}

/// An empty trait that only exists so that code that imports it still compiles.
///
/// The methods that this trait used to provide (`register`, `borrow_mut` and `reclaim`) are
/// inherent methods of [`OpaqueTracker`] now. Nothing implements the trait. It will be removed.
#[deprecated(note = "the methods are inherent methods of `OpaqueTracker` now; remove the import")]
pub trait Tracker<T> {}
