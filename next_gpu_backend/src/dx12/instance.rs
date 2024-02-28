use crate::instance::InstanceDescription;
use crate::instance::RawInstance;
pub struct DX12Instance {}

impl RawInstance for DX12Instance {}

impl DX12Instance {
    #[must_use]
    pub fn new(_desc: &InstanceDescription) -> Self {
        Self {}
    }
}
