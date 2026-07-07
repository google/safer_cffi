use std::marker::PhantomData;

/// A type-safe token representing an object held explicitly within an `OpaqueTracker`.
#[repr(transparent)]
pub struct Handle<T> {
    pub(crate) id: usize,
    _marker: PhantomData<T>,
}

// We cannot automatically derive these traits because of the phantom data.
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for Handle<T> {}
impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
impl<T> core::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Handle({:#x})", self.id)
    }
}

impl<T> Handle<T> {
    pub(crate) const HALF_BITS: u32 = usize::BITS / 2;
    pub(crate) const MAX_GENERATION: usize = (1usize << Self::HALF_BITS) - 1;
    pub(crate) const MAX_INDEX: usize = (1usize << Self::HALF_BITS) - 1;

    pub(crate) fn new(index: usize, generation: usize) -> Self {
        Self { id: (generation << Self::HALF_BITS) | index, _marker: PhantomData }
    }

    pub(crate) fn index(self) -> usize {
        self.id & Self::MAX_INDEX
    }

    pub(crate) fn generation(self) -> usize {
        self.id >> Self::HALF_BITS
    }

    /// Creates a null handle (essentially a null pointer).
    pub fn null() -> Self {
        Self { id: 0, _marker: PhantomData }
    }

    /// Converts the handle into a type-safe FFI pointer for C bridging.
    pub fn into_opaque_ptr(self) -> *mut T {
        // Do not implement From<Handle<T>> for *mut T, we want to force the conversion to be
        // explicit because it's not a normal but an opaque pointer.
        self.id as *mut T
    }
}

impl<T> From<*const T> for Handle<T> {
    fn from(ptr: *const T) -> Self {
        Self { id: ptr as usize, _marker: PhantomData }
    }
}

impl<T> From<*mut T> for Handle<T> {
    fn from(ptr: *mut T) -> Self {
        Self { id: ptr as usize, _marker: PhantomData }
    }
}
