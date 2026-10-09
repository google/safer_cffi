---
name: safer-cffi
description: >-
  Implements a C API in Rust with the safer_cffi crate while keeping the C ABI:
  crate structure (safe core vs. thin FFI layer, naming), opaque object handles
  (OpaqueTracker, Handle), nullable-pointer parameter idioms (Option-wrapped
  references, boxes and CStrRef), pointer+length array fields in repr(C) structs
  (CBufPtr, OwnedCBufPtr, CVecRefMut) backed by the C allocator, and matching C
  behaviour (out-params on error paths, aliased arguments, panics). Use when
  designing or porting a Rust drop-in replacement of a C library, writing or
  reviewing no_mangle extern "C" functions, repr(C) structs, or
  create/borrow/destroy object lifecycles, when choosing between raw pointers
  and safer_cffi types, or when fixing AbiCompatibleWith signature-test errors.
  Don't use for Blaze/BUILD wiring or differential testing setup of such a port
  (use safer-cffi-build-rules) or for calling an existing C library from Rust.
metadata:
  icon: 🦀
---

# safer_cffi: Writing a C API in Rust

`safer_cffi` lets `extern "C"` entry points be *safe* Rust functions: null
checks, ownership, and lifetimes are expressed in the signature, and the few
remaining `unsafe` blocks are confined to struct accessors with a one-line
invariant. Reach for it whenever a C API is being reimplemented in Rust.

Working examples for every pattern live in the crate's `examples/` directory
(`opaque_tracker`, `raw_tracker`, `functions`, `c_buf_ptr`,
`c_buf_ptr_with_capacity`). Copy from them rather than from memory.

## Crate layout

Keep the algorithm in plain Rust and the C ABI in a thin layer around it:

Module                                 | Contents                                                                                                                  | `unsafe`
:------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ | :-------
`c_types.rs` (often bindgen-generated) | `#[repr(C)]` structs and constants from the header, plus their buffer accessors (see below)                               | Only inside the accessors, each with `// SAFETY:`
`ffi.rs`                               | `#[unsafe(no_mangle)] pub extern "C" fn` wrappers: unpack arguments, call the core, map results to the C error convention | Ideally none; sometimes necessary for raw `(ptr, len)` parameters
Core modules (`decoder.rs`, ...)       | The algorithm, written against slices, references and Rust types                                                          | None: `#![forbid(unsafe_code)]` at the top of each file

*   **Size the FFI layer to the library.** For small and medium-sized libraries,
    one `ffi.rs` plus an optional `c_types.rs` is enough. Larger libraries can
    split `ffi.rs` into submodules (e.g. `ffi/decode.rs`, `ffi/encode.rs`).
*   **Naming:** the core follows Rust conventions (`d_gif_open_file`,
    `snake_case` locals). Only the `extern "C"` wrappers, and the `#[repr(C)]`
    structs/fields, keep the exact C names (`DGifOpenFileName`). Put
    `#![allow(non_snake_case)]` on the FFI modules and
    `#![allow(nonstandard_style)]` on the C types module (C type names like
    `z_stream` also trip `non_camel_case_types`), never on the crate root or the
    core.
*   **Wrappers stay thin.** No algorithm logic in the FFI layer: if a wrapper
    grows beyond argument unpacking and error mapping, move the logic into the
    core where `forbid(unsafe_code)` applies.
*   **Re-export at the crate root** (`pub use c_types::*; pub use ffi::*;`); the
    signatures test looks up every function and type there. A split `ffi` must
    also re-export its submodules (`pub use decode::*;` inside `ffi`), because
    `pub use ffi::*` does not reach into them.

## Choosing the Rust type for a C construct

**Encode the ownership assumptions that C leaves unstated.** A C `T*` looks the
same whether the callee borrows the object for the call, takes it over and frees
it, or reads it as the start of an array whose length lives elsewhere; the
contract exists only in comments and in what the C code does. We want to make
this explicit in the Rust API and use more expressive Rust types instead of raw
pointers. They are ABI compatible with the C type (no casts at the boundary),
but state assumptions that the compiler then enforces: references cannot escape
the call, `&mut` is exclusive, `Box`/`CBox` are freed exactly once by their own
allocator, and the object behind a `Handle` belongs to Rust alone.

Determine the ownership from what the C code does (implementation, callers,
docs), not from the pointer type: does the function free the pointer, keep it
after returning, or only use it during the call? Who allocated it, and with what
allocator? The signatures test cannot check this choice, since `Option<Box<T>>`
and `Option<&mut T>` both pass for a `T*`; the wrong one leaks the object or
frees it out from under the caller. Where no type can express an assumption, use
a pointer and write it down instead: in the function's `# Safety` section (see
[Parameters](#parameters-let-the-signature-do-the-null-checks)) or in the struct
field's safety-invariant comment (see below).

C construct                             | Rust spelling                                   | Notes
:-------------------------------------- | :---------------------------------------------- | :----
Opaque `typedef struct Foo Foo; Foo*`   | `Handle<Foo>`                                   | Managed by an `OpaqueTracker<Foo>`. Not `Option`: a null handle is `Handle::null()`.
`const T*` parameter                    | `Option<&T>`                                    | `None` is NULL. Also `Option<&[T; N]>` for fixed-size arrays. Not if the call frees or moves what it may point to, see [Aliased arguments](#aliased-arguments).
`T*` parameter, caller keeps ownership  | `Option<&mut T>`                                | Also `Option<&mut [T; N]>`.
`T*` out-parameter, possibly uninit     | `Option<&mut T>`                                | `*p = v`; never read before writing. If `T` has drop glue, use `Option<&mut MaybeUninit<T>>` + `p.write(v)`. Write it on the same paths as C, see [Out-parameters](#out-parameters-and-error-paths).
`T*` whose ownership moves to Rust      | `Option<Box<T>>` / `Option<CBox<T>>`            | `Box` if *Rust* allocated it with `Box`; `CBox` if allocated with `malloc`/`CBox`.
`const char*` parameter                 | `Option<CStrRef<'_>>`                           | `.to_c_str()` / `.to_bytes()`. Never `&CStr` or `&str` (fat pointers, wrong ABI).
`T* items; int len;` struct fields      | `OwnedCBufPtr<T>` (`T: Copy`) or `CBufPtr<T>`   | Plus the original `len` field, see below.
`T** items, int* len` parameters        | `Option<&mut CBufPtr<T>>`, `Option<&mut c_int>` | C-owned array Rust appends to, see below.
`T*` field pointing at one Rust object  | `Option<Box<T>>` / `Option<CBox<T>>`            | Dropped automatically with the parent struct.
`T*` returned to C that C will `free()` | `Option<CBox<T>>` / `CBufPtr::clone_and_leak`   | Both allocate with `malloc` so C's `free` is correct.

Return types follow the same rows: `-> Handle<Foo>`, `-> Option<Box<T>>`, `->
i32`, ... Use exact-width types that match the header (`i32` for `int32_t`,
`core::ffi::c_int` for `int`); a `u32` vs `int32_t` mismatch is an ABI bug even
though it links.

References (`Option<&T>`, `Option<&mut T>`) are for *parameters only*. Struct
fields need owning or raw types because they outlive the call.

## Opaque objects: `OpaqueTracker` + `Handle`

This is the default way to hand a Rust object to C. The handle is a generational
ID, not an address, so use-after-free, double-free and double-borrow become
`Err`s instead of memory corruption.

```rust
use safer_cffi::{Handle, OpaqueTracker, Tracker}; // `Tracker` trait must be in scope

pub struct Counter(u64);
static TRACKER: OpaqueTracker<Counter> = OpaqueTracker::new();

#[unsafe(no_mangle)]
pub extern "C" fn new_counter() -> Handle<Counter> {
    TRACKER.register(Box::new(Counter(0))).unwrap_or_else(|_| Handle::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn increase_counter(h: Handle<Counter>) -> u64 {
    let Ok(mut c) = TRACKER.borrow_mut(h) else { return 0 }; // NotFound / AlreadyBorrowedMutably
    c.0 += 1;
    c.0
} // guard drops here and returns the object to the tracker

#[unsafe(no_mangle)]
pub extern "C" fn free_counter(h: Handle<Counter>) {
    let _ = TRACKER.reclaim(h); // Err for stale/null/borrowed handles; nothing to do
}
```

Rules that follow from how the tracker works:

*   **Never `unwrap()` tracker results in an `extern "C"` fn.** A panic in an
    `extern "C"` function aborts the process. Map `TrackerError` to the C API's
    error convention (`0`, `-1`, `NULL`, `Handle::null()`). If the API must
    survive arbitrary panics, wrap the body in `std::panic::catch_unwind`.
*   **`borrow_mut` is exclusive, not a lock.** It moves the object out of the
    tracker; a second `borrow_mut` or a `reclaim` of the same handle while the
    guard is alive returns `AlreadyBorrowedMutably` instead of blocking. So:
    drop the `Tracked` guard *before* invoking C callbacks or calling other
    entry points that may touch the same handle (extract what you need first).
    Inside one entry point, borrow once and pass `&mut *guard` to helpers
    instead of re-borrowing the handle. Different handles never contend beyond a
    short internal mutex.
*   **One plain `static` tracker per Rust type.** `OpaqueTracker::new()` is
    `const`, so no `OnceLock`/`lazy_static` wrapper is needed. `register` fails
    with `CapacityExceeded` only after ~4 billion live objects.
*   **No explicit null check.** `Handle::null()` never matches a live entry, so
    `borrow_mut`/`reclaim` already return `Err(NotFound)` for it.
*   **Pointer fields in C structs that Rust owns** (e.g. `struct foo_state*
    state`) don't need a tracker: make the field `Option<Box<State>>`.
*   **Structs that C partially initialises** (zlib's `z_stream`: the caller sets
    `zalloc`, `next_in`, ... and leaves `state` as garbage) do need a tracker,
    because reading or dropping a garbage `Option<Box<_>>` is UB. Make such
    fields `Handle<State>` (`#[repr(transparent)]` over `usize`, so any bit
    pattern is valid): assign the result of `register` on init and
    `borrow_mut(strm.state)` afterwards. Garbage and stale values come back as
    `Err`. The same applies if C may copy the struct by value.
*   **`Handle<T>` is only for types C cannot see into.** If the header defines
    the struct's fields, C may read them through the pointer, so the Rust side
    must expose a real `#[repr(C)]` object (`Option<&mut T>` /
    `Option<Box<T>>`). Using `Handle<T>` there fails the signatures test.

### When to use `RawTracker` instead

`RawTracker<T>` keys objects by their real address (`*mut T`), so C can read
fields directly while Rust still rejects pointers it never handed out. Use it
only when the header exposes the struct *and* C needs direct field access.
Caveats: every object must be created by Rust (C-built structs are rejected),
freed+reallocated addresses silently resolve to the new object (ABA), and it is
measurably slower than `OpaqueTracker` (`benches/RESULTS.md`).

## Parameters: let the signature do the null checks

```rust
#[unsafe(no_mangle)]
pub extern "C" fn read_value(ptr: Option<&i32>) -> i32 { ptr.copied().unwrap_or(0) }

#[unsafe(no_mangle)]
pub extern "C" fn write_value(out: Option<&mut i32>, value: i32) {
    if let Some(p) = out { *p = value; }
}

#[unsafe(no_mangle)]
pub extern "C" fn take_ownership(obj: Option<Box<MyObj>>) { drop(obj); }

#[unsafe(no_mangle)]
pub extern "C" fn set_name(name: Option<CStrRef<'_>>) -> i32 {
    let Some(name) = name else { return -1 };
    let name = name.to_c_str().to_string_lossy(); // borrow is tied to the call
    ...
}
```

Keep the function *safe* (`pub extern "C" fn`, not `pub unsafe extern "C" fn`)
whenever every parameter is one of the safe spellings above and none depends on
another. Make it `unsafe extern "C" fn` with a `# Safety` section when soundness
rests on the caller:

*   `(ptr, len)` *parameter* pairs that the table cannot express. Do the single
    `unsafe` conversion at the top with a `// SAFETY:` comment naming the C
    contract, and pass slices/references to safe code. Such a pair can also be
    viewed through `CBufPtr`: `unsafe { CBufPtr::from_raw(p) }` then
    `.with_len(len)`.
*   Parameters that are only valid together, such as an array slot and its count
    (`Option<&mut CBufPtr<T>>` + `Option<&mut c_int>`, see below). Each spelling
    is safe on its own, but a safe Rust caller could pass a count that doesn't
    match the array.
*   Pointers that the call itself may invalidate, see
    [Aliased arguments](#aliased-arguments).

Struct accessors (below) stay safe: they rely on the struct's documented field
invariant instead.

## Struct fields: `(T* items, int len)` pairs

Replace the pointer with a `#[repr(transparent)]` wrapper and keep the integer
field exactly as the header declares it (`c_int`, `u32`, `usize`, ...).
Centralise the invariant in accessors (`items`, `items_mut`, `items_vec`) — only
define the ones the crate actually calls:

```rust
use core::ffi::c_int;
use safer_cffi::{CBufPtr, CVecRefMut, OwnedCBufPtr};

#[repr(C)]
pub struct IntArray {
    // Safety invariant: `items` points to exactly `item_len` initialised i32s
    // allocated with malloc (or is null and item_len == 0).
    pub items: OwnedCBufPtr<i32>,
    pub item_len: c_int,
}

impl IntArray {
    pub fn items(&self) -> &[i32] {
        // SAFETY: struct invariant (see field comment).
        unsafe { self.items.with_len(self.item_len) }
    }
    pub fn items_mut(&mut self) -> &mut [i32] {
        // SAFETY: struct invariant.
        unsafe { self.items.with_len_mut(self.item_len) }
    }
    // Only needed when the buffer grows, shrinks, swaps, or needs a Drop::clear():
    pub fn items_vec(&mut self) -> CVecRefMut<'_, i32, c_int> {
        // SAFETY: struct invariant; the handle keeps ptr and len in sync.
        unsafe { self.items.as_vec_mut(&mut self.item_len) }
    }
}
```

*   **`OwnedCBufPtr<T>` when `T: Copy`.** Always prefer `OwnedCBufPtr<T>` for
    owned buffers of `Copy` elements (e.g. `u8` byte buffers). It automatically
    frees the buffer on drop without needing the length, so it does not require
    a `Drop` impl that `clear()`s it. Build one from a `CVec<T>` (`Vec<T,
    LibcAlloc>`) with `OwnedCBufPtr::from_boxed_slice(v.into_boxed_slice())`, or
    copy a slice with `OwnedCBufPtr::clone_from_slice(src)`
    (`try_clone_from_slice` on fallible paths), and set the length field to the
    element count.
*   **`CBufPtr<T>` when elements need dropping (`T: !Copy`)** (e.g. an array of
    structs that themselves own nested buffers). Because `Drop` must run on each
    element before freeing the array, `OwnedCBufPtr` cannot be used; add `impl
    Drop { fn drop(&mut self) { self.items_vec().clear(); } }` and, if needed,
    `impl Clone` using `CBufPtr::clone_and_leak(self.items())`.
*   **Mutate through `CVecRefMut`, never by hand.** `push_back`,
    `try_push_back`, `clear`, `swap` keep `ptr`, `len` (and `cap`) consistent
    and use `malloc`/`realloc`/`free` so C can keep freeing with `free()`.
    Writing the length field directly or `realloc`ing yourself is how counts
    drift. `push_back` panics on OOM (an abort across `extern "C"`); use
    `try_push_back` when the C API reports allocation failure.
*   **Without a capacity field every `push_back` reallocates** (the allocation
    is always exactly `len` elements). If the C struct has a capacity field use
    `as_vec_mut_with_cap(&mut len, &mut cap)` to get amortised growth; otherwise
    build a `CVec` and convert it once with `from_boxed_slice`. Pick one of
    `as_vec_mut` / `as_vec_mut_with_cap` per field and never mix them: the
    former assumes the allocation is exactly `len`.
*   **Length types are checked at runtime.** Negative lengths panic ("CBufPtr:
    len is negative"), so validate C-supplied counts before storing them into
    the struct if C can pass garbage. `as_vec_mut` also panics on a null pointer
    with a non-zero length. A struct in either state is already corrupt: let it
    panic instead of patching the count (see [Panics](#panics)).
*   **`T** items, int* len` out-parameters** (C owns an array and asks Rust to
    append): take the slot directly as `Option<&mut CBufPtr<T>>`. The function
    must be `unsafe` because it trusts `*len` to match `*items`:

    ```rust
    /// # Safety
    ///
    /// `*items` must be a malloc-allocated array of `*len` initialised Items, or null with `*len == 0`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn add_item(items: Option<&mut CBufPtr<Item>>, len: Option<&mut c_int>, item: Item) -> c_int {
        let (Some(items), Some(len)) = (items, len) else { return ERROR };
        // SAFETY: guaranteed by the caller, see `# Safety`.
        match unsafe { items.as_vec_mut(len) }.try_push_back(item) {
            Ok(()) => OK,
            Err(_) => ERROR,
        }
    }
    ```

## Allocation across the boundary

Memory that C will `free()` must come from `malloc`; memory Rust `Box`es must
come from the Rust allocator (which need not be libc's, e.g. under jemalloc or
mimalloc). `safer_cffi` defaults everything C-facing to `LibcAlloc`:

*   `CBox<T>` (`Box<T, LibcAlloc>`): `CBox::new_in(v, LibcAlloc)`; use
    `Option<CBox<T>>` directly in `extern "C"` signatures and `#[repr(C)]`
    struct fields (or `CBox::into_raw(b)` / `CBox::from_raw_in(p, LibcAlloc)`
    when converting to/from `*mut T`).
*   `CVec<T>` (`Vec<T, LibcAlloc>`): `CVec::new_in(LibcAlloc)`, build it up with
    the usual `Vec` API, then hand it over with
    `CBufPtr`/`OwnedCBufPtr::from_boxed_slice(v.into_boxed_slice())`.
*   Buffers whose size C controls (e.g. `Width * Height`): reserve before
    filling, so that a huge size is an error code rather than an abort:
    `v.try_reserve_exact(n).map_err(..)?; v.resize(n, 0);`.
*   `CBufPtr::clone_and_leak(&[T])` for malloc-backed arrays (including
    nul-terminated `c_char` strings). `try_clone_and_leak` for fallible paths.
*   A custom `Allocator` is supported via the `*_in` variants (`as_vec_mut_in`,
    `clone_and_leak_in`) and `OwnedCBufPtr<T, A: DropByPtrAllocator>`; pass the
    *same* allocator instance every time.

## Matching C behaviour

The signatures test checks types, not behaviour. Ports drift from C in the
places below even when every type is right.

### Out-parameters and error paths

A reference's validity does not depend on the bytes it points to
([UCG#414](https://github.com/rust-lang/unsafe-code-guidelines/issues/414);
t-opsem consensus, not yet in the Reference or `std` docs), so a `&mut T` to
uninitialised memory is fine; only reading it as `T` is UB. Take out-params as
`Option<&mut T>` and assign with `*p = v`, but never read before the first write
(`*p += 1`, `if *p == 0`, `mem::replace(p, v)`), in the wrapper or in any core
function the reference reaches. Nothing enforces this, so note it in the core
function's doc comment.

If `T` has drop glue (e.g. `Option<CBox<U>>`), `*p = v` drops the garbage first:
use `Option<&mut MaybeUninit<T>>` and `p.write(v)` instead. Uninitialised output
buffers can be `&mut [T]` too; since `slice::from_raw_parts_mut` and
`CBufPtr::with_len_mut` still document initialised elements, cite UCG#414 in
that call's `// SAFETY:` comment.

C often stores an out-param *and then* fails in a later step (e.g. `*ExtCode =
Buf[0];` followed by a failing read of the first sub-block), or fails *before*
storing and leaves the caller's variable untouched. Passing `&mut T` into the
core and writing it where C does preserves both behaviours: returning the value
in `Ok(..)` loses it on later failure, splitting the core function around the
write breaks up the C structure, and pre-initialising `*ext_code = 0` in the
wrapper clobbers the caller's variable on early failure.

```rust
// Core: the same steps as C.
pub fn get_extension(gif: &mut GifFile, ext_code: &mut c_int) -> Result<bool, GifError> {
    let code = read_byte(gif)?;    // failing here leaves `*ext_code` untouched
    *ext_code = c_int::from(code); // stored even if the next read fails, like C
    get_extension_next(gif)
}

// Wrapper: pass `&mut c_int` straight through.
let result = decoder::get_extension(gif, ext_code);
```

Also write what C writes on success (e.g. `*ErrorCode = D_GIF_SUCCEEDED` in a
close function), and comment deliberate differences (e.g. reporting an error
code where C leaves `*ErrorCode` untouched).

### Aliased arguments

A reference parameter must stay valid until the function returns, even if it is
not used again: freeing or reallocating what it points to during the call is
undefined behaviour. C callers routinely pass a struct's own field back in, e.g.
`EGifPutScreenDesc(gif, ..., gif->SColorMap)` or `GifMakeSavedImage(gif,
&gif->SavedImages[0])`. If an entry point replaces, frees or grows something a
parameter may point into:

*   **Replace:** take the parameter as a raw `*const T` (making the function
    `unsafe`). Define a dual-state `Arg<T>` enum:

    ```rust
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Arg<T> {
        Current,
        New(T),
    }
    ```

    In the `unsafe extern "C"` wrapper, use `Arg::Current` if the parameter
    points to the stored field; otherwise make an immediate copy or dereference:

    ```rust
    let arg = if obj.field.as_deref().is_some_and(|f| ptr::eq(f, raw_ptr)) {
        Arg::Current
    } else {
        Arg::New(unsafe { raw_ptr.as_ref() })
    };
    ```

    If the core clones `Arg::New` values, clone at the FFI boundary and accept
    `Arg<T>` instead of `Arg<&T>`. This makes it easier to reason about aliasing
    violations: `Arg::New(unsafe { raw_ptr.as_ref() }.cloned())`. Don't compare
    addresses in the core. Internal callers pass `Arg::Current` or an owned
    value or reference without aliasing violations.
*   **Grow:** take the parameter as a raw `*const T` (so the function becomes
    `unsafe`), copy from it with `unsafe { p.as_ref() }` *before* calling into
    the core.

### Panics

A panic in an `extern "C"` function aborts the process where C returns an error.
On data C controls (arguments, file contents, callback results, field values),
treat every `unwrap`, `expect`, `slice[i]`, `try_into().expect(..)` and signed
`as usize` as a bug: use `.get()`/`.get_mut()`, `first_chunk::<N>()` and
`try_from`, and map the failure to the C error code. Allocations whose failure C
reports need the `try_*` APIs.

The exception is a struct that breaks its own safety invariant, such as a null
array with a non-zero count or a negative count. It is already corrupt, and C
only copes with it by accident. Let the `CBufPtr` checks panic.

## Anti-patterns

Don't                                                                                | Do instead                                                                           | Why
:----------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :--
`pub unsafe extern "C" fn f(p: *mut T)` + `if p.is_null()` + `&mut *p`               | `pub extern "C" fn f(p: Option<&mut T>)`                                             | Same ABI, no `unsafe`, null handled by the type.
`Box::into_raw` / `Box::from_raw` for opaque objects                                 | `OpaqueTracker` + `Handle<T>`                                                        | `from_raw` on a stale or forged pointer is UB; the tracker returns `Err`.
`&CStr`, `&str`, `String` in an `extern "C"` signature                               | `Option<CStrRef<'_>>` (input) / `clone_and_leak` of bytes (output)                   | Rust strings are fat pointers; not a `char*`.
`Option<Box<T>>` for a pointer C may have `malloc`ed                                 | `Option<CBox<T>>`                                                                    | Allocator mismatch on drop.
`Handle<T>` for a struct whose fields appear in the header                           | `#[repr(C)] T` with `Option<&mut T>` / `Option<Box<T>>` (or `RawTracker`)            | C code reads the fields through the pointer.
`TRACKER.borrow_mut(h).unwrap()`                                                     | `let Ok(mut x) = TRACKER.borrow_mut(h) else { return ERR }`                          | Panics abort across `extern "C"`.
`if h == Handle::null() { return ERR }` before `borrow_mut`/`reclaim`                | Just handle the `Err`                                                                | A null handle never matches; the tracker already returns `NotFound`.
`static T: OnceLock<OpaqueTracker<_>>` / `lazy_static!`                              | `static T: OpaqueTracker<_> = OpaqueTracker::new();`                                 | `new()` is `const`.
Holding a `Tracked` guard while calling a C callback                                 | Copy out what the callback needs, drop the guard, then call                          | Re-entrant use of the same handle fails with `AlreadyBorrowedMutably`.
`Option<&mut MaybeUninit<T>>` + `p.write(v)` for an out-param without drop glue      | `Option<&mut T>` + `*p = v`, written where C writes it; never read it first          | `&mut T` to uninitialised memory is fine ([UCG#414](https://github.com/rust-lang/unsafe-code-guidelines/issues/414)); reading it as `T` is not, and `.write(0)` placeholders clobber `*p` on early errors.
`*mut T` + `len` fields with ad-hoc `from_raw_parts` at each use site                | `OwnedCBufPtr`/`CBufPtr` + accessors holding the only `unsafe`                       | One invariant, one place to audit.
Bumping `len` yourself after writing past the end                                    | `items_vec().push_back(v)`                                                           | The handle reallocates and updates `len` atomically.
`push_back(v)` followed by `count += 1` (or any helper that already grows the array) | Let `CVecRefMut` own the count                                                       | The extra increment leaves an uninitialised ghost element that C later reads.
`n as usize` on a signed C return or callback result (`-1` = error/EOF)              | Check `n < 0` first, or `usize::try_from(n)`                                         | `-1` becomes `usize::MAX` and slips past guards like `n < 1`.
C-style names (`DGifGetLine`) or `#![allow(non_snake_case)]` in core modules         | Rust names in the core; C names only on `extern "C"` wrappers and `#[repr(C)]` types | Keeps the lint useful where the logic lives.
`push_back` in an entry point that returns an error code on OOM                      | `try_push_back(v)` and map `Err` to the error code                                   | `push_back` panics on allocation failure.
`impl Drop` that calls `clear()` on an `OwnedCBufPtr` field                          | Nothing; `OwnedCBufPtr` frees itself                                                 | Redundant `unsafe` and a second invariant to keep in sync.
`OwnedCBufPtr::from_boxed_slice(std_vec.into_boxed_slice())`                         | Build a `CVec` (`Vec<T, LibcAlloc>`)                                                 | C frees the buffer with `free()`; it must come from `malloc`.
`RawTracker` "because the API already passes pointers"                               | `OpaqueTracker` unless C dereferences the pointer                                    | Raw keys are slower and vulnerable to ABA reuse.
Mismatched integer widths (`u32` for `int32_t`, `usize` for `int`)                   | Match the header exactly (`i32`, `c_int`)                                            | Links fine, corrupts data on the C side.
Replacing or freeing a field while a parameter may point to it (`f(g, g->map)`)      | Raw pointer in wrapper; `Arg::Current` if it is that field, else `Arg::New(copy)`    | `take()` drops fields on early errors; a borrowed parameter may alias anything `g` owns, not just the replaced field.
Returning an out-param in `Ok(..)`, or splitting the core function around it         | Pass it into the core as `&mut T` and write it where C does                          | C may store it before a later step fails; the core keeps C's control flow.
Safe `extern "C" fn` that trusts one parameter to match another (slot + count)       | `unsafe extern "C" fn` with a `# Safety` section                                     | Safe Rust callers could pass mismatched values.
Resetting the count or returning early when the array is null but the count is not   | Let `as_vec_mut` panic                                                               | The struct is already corrupt, and C only copes with it by accident.
