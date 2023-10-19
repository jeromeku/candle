pub struct CLIPVisionModel;
pub struct CLIPImageProcessor;
pub struct CLIPVisionConfig;

pub struct CLIPVisionTower {
    vision_tower_name: String,
    select_layer: String,
    select_feature: String,
    image_processor: CLIPImageProcessor,
    vision_tower: CLIPVisionModel,
}
