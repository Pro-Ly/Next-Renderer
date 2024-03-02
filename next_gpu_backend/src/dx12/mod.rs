pub struct Api;

impl crate::Api for Api {
    type Instance = Instance;
}

pub struct Instance {}

impl crate::Instance<Api> for Instance {
    fn new(_desc: &crate::api::instance::Description) -> Self {
        Self {}
    }
}
