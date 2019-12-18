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

        impl From<$wrapper> for $wrapped {
            #[inline]
            fn from(outer: $wrapper) -> $wrapped {
                outer.0
            }
        }

        impl From<$wrapped> for $wrapper {
            #[inline]
            fn from(inner: $wrapped) -> $wrapper {
                $wrapper(inner)
            }
        }
    }
}

pub(crate) use typed_resource_wrapper;
