# Building Blocks for Safer C FFI

`safer_cffi` provides Rust primitives for replacing C libraries with memory-safe
Rust implementations while maintaining full ABI compatibility with the original
C headers.

*   **[Pointer Trackers](#pointer-trackers)** — Manage the lifecycle of Rust
    objects handed to C as opaque handles or raw pointers, preventing
    use-after-free and double-borrow bugs.
*   **[Struct Field Helpers](#struct-field-helpers)** — Safe wrappers for common
    C struct field patterns (`*mut T`/`c_int` array pairs and `const char*`
    strings).

**Tip:** Take a look at [`examples/`](examples/) for patterns and common use
cases.

## Pointer Trackers

Trackers manage the lifecycle of Rust objects that are handed to C code as
opaque handles or pointers. They ensure objects are only accessed when valid and
prevent concurrent mutable access.

Both trackers implement the `Tracker<T>` trait, which provides:

*   `register(Box<T>)` → Key — stores the object, returns a key for C.
*   `borrow_mut(key)` → `Tracked<T>` — exclusive access via a RAII guard.
*   `reclaim(key)` → `Box<T>` — takes back ownership, removing from tracker.

### `OpaqueTracker<T>` (recommended)

Uses generational `Handle<T>` IDs. Prevents use-after-free via generation checks
and detects double-borrows.

```rust
static TRACKER: safer_cffi::OpaqueTracker<MyObj> = safer_cffi::OpaqueTracker::new();

#[unsafe(no_mangle)]
pub extern "C" fn create() -> Handle<MyObj> {
    TRACKER.register(Box::new(MyObj::default())).unwrap_or_else(|_| Handle::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn destroy(h: Handle<MyObj>) {
    let _ = TRACKER.reclaim(h);
}
```

See [`examples/opaque_tracker/`](examples/opaque_tracker/) for a full example.

### `RawTracker<T>`

Uses raw memory addresses (`*mut T`) as keys. Use only when C code needs the
actual pointer value (e.g. for direct field access). Caveats:

*   Cannot be used if C creates objects — all objects must originate from Rust.
*   Does not prevent ABA problems: a freed and re-allocated address silently
    resolves to the new object.

See [`examples/raw_tracker/`](examples/raw_tracker/) for a full example.

## Struct Field Helpers

### C Slices — `CSlicePtr<T>`

Many C structs contain `(*mut T, c_int)` pairs representing dynamically-sized
arrays. `CSlicePtr` can be used in place of `*mut T` and provides a safe handle
for access and manipulation.

*   **`CSlicePtr<T>`**:

    *   `with_len(c_int)` → `&[T]` — shared slice view.
    *   `with_len_mut(&mut c_int)` → `CSliceRefMut` — mutable handle.
    *   `clone_and_leak(&[T])` → `CSlicePtr<T>` — create a new CSlicePtr by
        cloning an existing slice.

*   **`CSliceRefMut<'a, T>`**: A borrowed mutable handle. Implements `DerefMut`
    to `&mut [T]`. Additional methods:

    *   `add(T)` — append via `realloc`.
    *   `clear()` — drop all elements, free memory, reset to null/0.
    *   `swap(&mut CSliceRefMut)` — swap two handles.

Usage example:

```rust
#[repr(C)]
struct MyStruct {
    // Safety invariant: the length of this array is `item_count`.
    items: CSlicePtr<Item>,
    item_count: c_int,
}

impl MyStruct {
    fn items(&self) -> &[Item] {
        // SAFETY: the length of `items` is `item_count`.
        unsafe { self.items.with_len(self.item_count) }
    }
    fn items_mut(&mut self) -> CSliceRefMut<'_, Item> {
        // SAFETY: the length of `items` is `item_count`.
        unsafe { self.items.with_len_mut(&mut self.item_count) }
    }
}

impl Drop for MyStruct {
    fn drop(&mut self) {
        self.items_mut().clear();
    }
}
```

See [`examples/c_slice_ptr/`](examples/c_slice_ptr/) for a full example.

### C Strings — `CStrRef<'a>`

A `#[repr(transparent)]` wrapper around `NonNull<c_char>` that can be used as
`Option<CStrRef<'_>>` in FFI signatures where a nullable C string is expected.
Unlike `core::ffi::CStr`, it guarantees a thin pointer layout and ABI
compatibility with a C `const char*`.

Usage example:

```rust
use safer_cffi::CStrRef;

#[unsafe(no_mangle)]
pub extern "C" fn print_string(s: Option<CStrRef<'_>>) {
    if let Some(c_str) = s {
        // CStrRef can be safely converted to a &CStr
        println!("Received: {}", c_str.to_c_str().to_string_lossy());
    }
}
```
