use std::mem::ManuallyDrop;

/// A Tag which contains either an allocated String or a `&'static str`
#[derive(Debug, Clone)]
pub enum Tag {
    /// An allocated String.
    Allocated(String),
    /// A static reference to a string.
    Static(&'static str)
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
         match self {
            &Tag::Allocated(ref tag) => write!(f, "{}", tag),
            &Tag::Static(ref tag) => write!(f, "{}", tag),
        }
    }
}

/// This type, and structs containing this type, must explicitly be destroyed
/// rather than simply being Dropped. Being Dropped will cause a panic.
#[derive(Debug)]
pub struct NoDrop(ManuallyDrop<Tag>);

impl NoDrop {
    /// Create a new NoDrop from a Tag
    pub fn new(tag: Tag) -> Self {
        Self(ManuallyDrop::new(tag))
    }

    /// Create a new NoDrop from an allocated String
    pub fn from_string<S: Into<String>>(tag: S) -> Self {
        Self(ManuallyDrop::new(Tag::Allocated(tag.into())))
    }

    /// Create a new NoDrop from an `&'static str`
    pub fn from_str(tag: &'static str) -> Self {
        Self(ManuallyDrop::new(Tag::Static(tag)))
    }

    /// Destroy this `NoDrop`
    pub fn destroy(mut self) {
        unsafe { ManuallyDrop::drop(&mut self.0) };
        core::mem::forget(self);
    }
}

impl Default for NoDrop {
    fn default() -> Self {
        Self::from_str("anonymous")
    }
}

impl Drop for NoDrop {
    fn drop(&mut self) {
        panic!("NoDrop item with tag {} was dropped!", *self.0);
    }
}