// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::handle::Handle;
use crate::opaque::OpaqueTracker;
use std::ops::{Deref, DerefMut};

/// A guard that provides mutable access to a tracked object.
///
/// The object is returned to the tracker when this guard is dropped.
pub struct Tracked<'tracker, T> {
    tracker: &'tracker OpaqueTracker<T>,
    key: Handle<T>,
    val: Option<Box<T>>,
}

impl<'tracker, T> Tracked<'tracker, T> {
    pub(crate) fn new(tracker: &'tracker OpaqueTracker<T>, key: Handle<T>, val: Box<T>) -> Self {
        Self { tracker, key, val: Some(val) }
    }
}

impl<T> Deref for Tracked<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.val.as_ref().unwrap()
    }
}

impl<T> DerefMut for Tracked<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.val.as_mut().unwrap()
    }
}

impl<T> Drop for Tracked<'_, T> {
    fn drop(&mut self) {
        let v = self.val.take().expect("corrupted Tracked state: value is None");
        self.tracker.return_mut(self.key, v);
    }
}

/// An empty trait that only exists so that code that imports it still compiles.
///
/// The methods that this trait used to provide (`register`, `borrow_mut` and `reclaim`) are
/// inherent methods of [`OpaqueTracker`] now. Nothing implements the trait. It will be removed.
#[deprecated(note = "the methods are inherent methods of `OpaqueTracker` now; remove the import")]
pub trait Tracker<T> {}
