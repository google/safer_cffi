// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use safer_cffi::Tracker;

#[derive(Default)]
pub struct Counter(u64);

impl Counter {
    pub fn increase(&mut self) {
        self.0 += 1;
    }
    pub fn get(&self) -> u64 {
        self.0
    }
}

static TRACKER: safer_cffi::OpaqueTracker<Counter> = safer_cffi::OpaqueTracker::new();

#[unsafe(no_mangle)]
pub extern "C" fn new_counter() -> safer_cffi::Handle<Counter> {
    let counter = Box::new(Counter::default());
    TRACKER.register(counter).unwrap_or_else(|_| safer_cffi::Handle::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn free_counter(counter: safer_cffi::Handle<Counter>) {
    let _ = TRACKER.reclaim(counter);
}

#[unsafe(no_mangle)]
pub extern "C" fn increase_counter(counter: safer_cffi::Handle<Counter>) -> u64 {
    let Ok(mut tracked) = TRACKER.borrow_mut(counter) else {
        return 0;
    };
    tracked.increase();
    tracked.get()
}
