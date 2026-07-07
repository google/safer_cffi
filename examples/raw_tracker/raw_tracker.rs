use safer_cffi::Tracker;

#[repr(C)]
pub struct MyStruct {
    pub field: u64,
}

static TRACKER: safer_cffi::RawTracker<MyStruct> = safer_cffi::RawTracker::new();

#[unsafe(no_mangle)]
pub extern "C" fn new_struct() -> *mut MyStruct {
    let s = Box::new(MyStruct { field: 0 });
    TRACKER.register(s).unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn free_struct(s: *mut MyStruct) {
    let _ = TRACKER.reclaim(s);
}

#[unsafe(no_mangle)]
pub extern "C" fn print_struct(s: *mut MyStruct) {
    // This will fail if `s` has not been created by `new_struct`.
    if let Ok(s) = TRACKER.borrow_mut(s) {
        println!("MyStruct field: {}", s.field);
    }
}
