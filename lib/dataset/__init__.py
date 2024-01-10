from ._plate import _plate


def get_dataset(config):
    if config.DATASET.DATASET == "plate":
        return _plate
    else:
        raise NotImplemented()
