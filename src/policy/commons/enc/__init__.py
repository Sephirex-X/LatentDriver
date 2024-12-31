from .bert import BertEncoder
__all__ = {
    'bert': BertEncoder,
}
def build_model(config):
    model = __all__[config.name](**config)
    return model