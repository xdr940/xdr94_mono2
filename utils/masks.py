

def VarMask(erro_maps):
    '''
    var mask
    :param erro_maps:
    :return:
    '''
    rhosvar = erro_maps.var(dim=1, unbiased=False)  # BHW
    rhosvar_flat = rhosvar.flatten(start_dim=1)  # B,H*W
    median, _ = rhosvar_flat.median(dim=1)  # b
    rhosvar_flat /= median.unsqueeze(1)
    delta_var = rhosvar_flat.mean(dim=1).unsqueeze(1)  # B
    # var_mask = (rhosvar_flat > 0.001).reshape_as(map_0)
    var_mask = (rhosvar_flat > delta_var / 10).reshape_as(rhosvar)

    return var_mask.float()
def MeanMask(erro_maps):
    '''
    mean mask
    :param erro_maps:
    :return:
    '''
    rhosmean = erro_maps.mean(dim=1)  # BHW
    rhosmean_flat = rhosmean.flatten(start_dim=1)  # b,h*w
    delta_mean = rhosmean_flat.mean(dim=1).unsqueeze(dim=1)  # b,1
    mean_mask = (rhosmean_flat < 2 * delta_mean).reshape_as(rhosmean)
    return mean_mask.float()
def IdenticalMask(erro_maps):
    pass