import torch


class _DataFactory:
    def __init__(self):
        self.datasets = dict()  # register_datasets()

    def register(self, config_class, data_class):
        self.datasets[config_class.__name__] = data_class

    def create(self, data_config) -> torch.utils.data.Dataset:
        # breakpoint()
        return self.datasets[data_config.__class__.__name__](data_config)

    def get_class(self, data_config):
        return self.datasets[data_config.__class__.__name__]


_data_factory = None


def register_dataset(config_class, dataset):
    global _data_factory
    _data_factory.register(config_class, dataset)


def get_factory():
    global _data_factory
    if _data_factory is None:
        from lib.data_registry import register_datasets

        _data_factory = _DataFactory()
        _data_factory.datasets = register_datasets()

    return _data_factory
