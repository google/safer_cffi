use crate::errors::TrackerError;
use std::ops::{Deref, DerefMut};

/// A trait for managing the lifecycle of Rust objects across FFI boundaries.
///
/// Trackers provide a way to store Rust objects and retrieve them using a key
/// that can be passed to C code. They ensure that objects are only
/// accessed when valid, and prevent concurrent mutable access errors.
pub trait Tracker<T> {
    /// The type of key used to identify tracked objects.
    type Key: Copy + Eq;

    /// Registers a new object with the tracker, taking ownership.
    ///
    /// Returns a `Key` that can be passed to C.
    fn register(&self, b: Box<T>) -> Result<Self::Key, TrackerError<Self::Key>>;

    /// Reclaims ownership of the object from the tracker.
    ///
    /// This removes the object from the tracker and returns the `Box<T>`.
    /// Fails if the object is currently borrowed.
    fn reclaim(&self, key: impl Into<Self::Key>) -> Result<Box<T>, TrackerError<Self::Key>>;

    /// Borrows the object, returning a guard that gives mutable access.
    ///
    /// Returns a `Tracked` guard that implements `DerefMut`. When the guard is dropped,
    /// the object is automatically returned to the tracker.
    ///
    /// Fails if the object is currently borrowed.
    fn borrow_mut(
        &self,
        key: impl Into<Self::Key>,
    ) -> Result<Tracked<'_, T, Self>, TrackerError<Self::Key>>
    where
        Self: Sized;

    /// Returns the borrowed object back to the tracker.
    /// This should only be called by the `Drop` implementation of the `Tracked` guard.
    #[doc(hidden)]
    fn return_mut(&self, key: Self::Key, val: Box<T>);
}

/// A guard that provides mutable access to a tracked object.
///
/// The object is returned to the tracker when this guard is dropped.
pub struct Tracked<'a, T, Tr: Tracker<T>> {
    tracker: &'a Tr,
    key: Tr::Key,
    val: Option<Box<T>>,
}

impl<'a, T, Tr: Tracker<T>> Tracked<'a, T, Tr> {
    pub fn new(tracker: &'a Tr, key: Tr::Key, val: Box<T>) -> Self {
        Self { tracker, key, val: Some(val) }
    }
}

impl<'a, T, Tr: Tracker<T>> Deref for Tracked<'a, T, Tr> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.val.as_ref().unwrap()
    }
}

impl<'a, T, Tr: Tracker<T>> DerefMut for Tracked<'a, T, Tr> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.val.as_mut().unwrap()
    }
}

impl<'a, T, Tr: Tracker<T>> Drop for Tracked<'a, T, Tr> {
    fn drop(&mut self) {
        let v = self.val.take().expect("corrupted Tracked state: value is None");
        self.tracker.return_mut(self.key, v);
    }
}
