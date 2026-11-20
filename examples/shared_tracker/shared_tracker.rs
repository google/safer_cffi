// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use safer_cffi::{Handle, OpaqueTracker};
use std::sync::Arc;

#[derive(Default)]
pub struct SharedCounter(u64);

impl SharedCounter {
    pub fn increase(&mut self) {
        self.0 += 1;
    }
    pub fn get(&self) -> u64 {
        self.0
    }
}

// The tracker holds `Arc<SharedCounter>` instead of the default `Box<SharedCounter>`, which
// enables shared borrows with `borrow`.
static TRACKER: OpaqueTracker<SharedCounter, Arc<SharedCounter>> = OpaqueTracker::new();

#[unsafe(no_mangle)]
pub extern "C" fn new_shared_counter() -> Handle<SharedCounter> {
    let counter = Arc::new(SharedCounter::default());
    TRACKER.register(counter).unwrap_or_else(|_| Handle::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn free_shared_counter(counter: Handle<SharedCounter>) {
    let _ = TRACKER.reclaim(counter);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_shared_counter(counter: Handle<SharedCounter>) -> u64 {
    // A shared borrow succeeds while other shared borrows of the same counter are alive.
    let Ok(tracked) = TRACKER.borrow(counter) else {
        return 0;
    };
    tracked.get()
}

#[unsafe(no_mangle)]
pub extern "C" fn increase_shared_counter(counter: Handle<SharedCounter>) -> u64 {
    // An exclusive borrow fails while any other borrow of the same counter is alive.
    let Ok(mut tracked) = TRACKER.borrow_mut(counter) else {
        return 0;
    };
    tracked.increase();
    tracked.get()
}
