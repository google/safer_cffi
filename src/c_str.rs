// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::ffi::{c_char, CStr};
use core::marker::PhantomData;
use core::ptr::NonNull;

/// A transparent wrapper around `*const c_char` that can be safely used as `Option<CStrRef<'_>>`
/// in FFI boundaries where a C string is expected.
///
/// Unlike `core::ffi::CStr`, `CStrRef` guarantees a thin pointer layout and ABI compatibility with a C pointer.
///
/// # Safety Invariants
/// The pointer must be non-null, properly aligned, and point to a null-terminated string
/// valid for the lifetime `'a`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CStrRef<'a> {
    ptr: NonNull<c_char>,
    _marker: PhantomData<&'a CStr>,
}

impl<'a> CStrRef<'a> {
    /// Returns the underlying CStr.
    pub fn to_c_str(self) -> &'a CStr {
        // SAFETY: The safety invariants of `CStrRef` guarantee that the pointer is valid,
        // properly aligned, and points to a null-terminated string valid for lifetime 'a.
        unsafe { CStr::from_ptr(self.ptr.as_ptr()) }
    }

    /// Converts this C string to a byte slice.
    ///
    /// The returned slice will **not** contain the trailing nul terminator that this C
    /// string has.
    pub fn to_bytes(self) -> &'a [u8] {
        self.to_c_str().to_bytes()
    }

    /// Creates a new `CStrRef` from a valid `CStr`.
    pub fn from_c_str(c_str: &'a CStr) -> Self {
        Self {
            ptr: NonNull::new(c_str.as_ptr() as *mut c_char).expect("CStr pointers are non-null"),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
        use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_c_str_ref_roundtrip() -> Result<()> {
        let c_str = c"hello";
        let cstr_ptr = CStrRef::from_c_str(c_str);
        let roundtrip = cstr_ptr.to_c_str();
        verify_that!(c_str, eq(roundtrip))?;
        verify_that!(roundtrip.as_ptr(), eq(c_str.as_ptr()))?;
        verify_that!(cstr_ptr.to_bytes(), eq(c_str.to_bytes()))?;
        Ok(())
    }
}
