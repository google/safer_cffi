// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

pub use allocator_api2::alloc::AllocError;
