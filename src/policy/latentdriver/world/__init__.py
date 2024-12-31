from .latent_world_model import LatentWorldModel
__all__ = {
    'latent_world_model': LatentWorldModel,
}
def build_model(config):
    model = __all__[config.name](**config)
    return model