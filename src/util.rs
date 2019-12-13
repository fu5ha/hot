macro_rules! typed_resource_wrapper {
    {
        $(#[$outer:meta])*
        pub struct $wrapper:ident($wrapped:ident);
    } => {
        $(#[$outer])*
        pub struct $wrapper($wrapped);

        impl $wrapper {
            /// Get the wrapped raw version of this resource.
            pub fn raw(&self) -> &$wrapped {
                &self.0
            }
        }
    }
}

pub(crate) use typed_resource_wrapper;
