# unified_ssl_3d/models/__init__.py
from networks.unet_3d import unet_3D, unet_3D_dt, unet_3D_dv_semi


def get_model(method: str, in_channels: int = 2, n_classes: int = 2):
    if method == "ugmcl":
        return unet_3D_dt(in_channels, n_classes)
    if method == "urpc":
        return unet_3D_dv_semi(in_channels, n_classes)
    return unet_3D(in_channels, n_classes)
