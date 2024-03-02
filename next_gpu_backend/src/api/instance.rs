use bitflags::bitflags;

pub struct Description {
    flags: CreationFlags,
}

impl Description {
    #[must_use]
    pub fn flags(&self) -> &CreationFlags {
        &self.flags
    }
}

impl Default for Description {
    #[must_use]
    fn default() -> Self {
        Self {
            flags: CreationFlags::empty(),
        }
    }
}

bitflags! {
    pub struct CreationFlags: u32 {
        const DEBUG = 1 << 0;
        const VALIDATION = 1 << 1;
    }
}
