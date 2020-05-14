
.1 3_mask with erodil
    min_{3,4} + f(M_{id}) + M_{var} + f(M_{mean})
```python


 # -------------------------
 
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            map1, idxs_1 = torch.min(reprojection_loss,dim=1)

            rhosvar = erro_maps.var(dim=1, unbiased=False)  # BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)  # B,H*W
            median, _ = rhosvar_flat.median(dim=1)  # b
            rhosvar_flat /= median.unsqueeze(1)
            delta_var = rhosvar_flat.mean(dim=1).unsqueeze(1)  # B

            #var_mask = (rhosvar_flat > 0.001).reshape_as(map_0)
            var_mask = (rhosvar_flat>delta_var/10).reshape_as(map_0)
            #两种效果基本完全相同,末端测试0.001白色稍微微多一些

            # rhosmean
            rhosmean = erro_maps.mean(dim=1)  # BHW
            rhosmean_flat = rhosmean.flatten(start_dim=1)#b,h*w
            delta_mean = rhosmean_flat.mean(dim=1).unsqueeze(dim=1)#b,1
            mean_mask = (rhosmean_flat <2* delta_mean).reshape_as(map_0)
            mean_mask0 = rectify(mean_mask)
            mean_mask_anti = 1 - mean_mask0
            mean_pixels = map1 *mean_mask_anti.float()

            # mean mask : 1 说明为moving region
            #ind_mov = (1 - var_mask) * mean_mask

            #static = (1 - var_mask) * (1 - mean_mask)
            #identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)
            identity_selection = (idxs_0 >= 2)#
            identity_selection2 = rectify(identity_selection)
            final_mask = var_mask.float()*mean_mask0.float()*identity_selection2.float()
            to_optimise = map1 * final_mask


            outputs["identity_selection/{}".format(scale)] = identity_selection.float()
            outputs["identity_selection2/{}".format(scale)] = identity_selection2.float()

            outputs["mean_mask/{}".format(scale)] = mean_mask.float()
            outputs["mean_mask0/{}".format(scale)] = mean_mask0.float()

            outputs["var_mask/{}".format(scale)] = var_mask.float()



            outputs["final_selection/{}".format(scale)] = final_mask.float()
            outputs["mean_pixels/{}".format(scale)] = mean_pixels.float()
            outputs["rhosmean/{}".format(scale)] = rhosmean.float()

            # ---------------------
```

.2 min_{3,4} + M_{id} 

```python

 # -------------------------
            map_1234, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            map_34, idxs_1 = torch.min(reprojection_loss,dim=1)

            
            identity_selection = (idxs_0 >= 2)#

            to_optimise = map_34 *  identity_selection.float()


            outputs["identity_selection/{}".format(scale)] = identity_selection.float()



            # ---------------------
```

.3 min_{3,4} + M_v 

```python

map_1234, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
map_34, idxs_1 = torch.min(reprojection_loss, dim=1)

var_mask  = VarMask(erro_maps)



final_mask = var_mask
to_optimise = map_34 * final_mask


outputs["var_mask/{}".format(scale)] = var_mask.float()

```



.4 min_{3,4} +M_{id} + M_v 

```python

map_1234, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
map_34, idxs_1 = torch.min(reprojection_loss, dim=1)

var_mask  = VarMask(erro_maps)


identity_selection = (idxs_0 >= 2).float() #

final_mask = var_mask*identity_selection

to_optimise = map_34 * final_mask


outputs["var_mask/{}".format(scale)] = var_mask.float()

```




.5 min_{3,4} + M_{id} + M_v + M_ean

```python

map_1234, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
map_34, idxs_1 = torch.min(reprojection_loss, dim=1)

var_mask  = VarMask(erro_maps)
mean_mask = MeanMask(erro_maps)


identity_selection = (idxs_0 >= 2).float() #

final_mask = var_mask* mean_mask * identity_selection
to_optimise = map_34 * final_mask

outputs["identity_selection/{}".format(scale)] = identity_selection.float()

outputs["mean_mask/{}".format(scale)] = mean_mask.float()

outputs["var_mask/{}".format(scale)] = var_mask.float()

outputs["final_selection/{}".format(scale)] = final_mask.float()

```




.6 min_{3,4} + M_{id} + M_v + f(M_ean)

```python

map_1234, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
map_34, idxs_1 = torch.min(reprojection_loss, dim=1)

var_mask  = VarMask(erro_maps)
mean_mask = MeanMask(erro_maps)
mean_mask = rectify(mean_mask)


identity_selection = (idxs_0 >= 2).float() #

final_mask = var_mask* mean_mask * identity_selection
to_optimise = map_34 * final_mask

outputs["identity_selection/{}".format(scale)] = identity_selection.float()

outputs["mean_mask/{}".format(scale)] = mean_mask.float()

outputs["var_mask/{}".format(scale)] = var_mask.float()

outputs["final_selection/{}".format(scale)] = final_mask.float()

```



min_{3,4} + f(M_{id}) + M_v + f(M_ean)

```python

map_1234, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
map_34, idxs_1 = torch.min(reprojection_loss, dim=1)

var_mask  = VarMask(erro_maps)
mean_mask = MeanMask(erro_maps)
mean_mask = rectify(mean_mask)


identity_selection = (idxs_0 >= 2).float() #
identity_selection = rectify(identity_selection)

final_mask = var_mask* mean_mask * identity_selection
to_optimise = map_34 * final_mask

outputs["identity_selection/{}".format(scale)] = identity_selection.float()

outputs["mean_mask/{}".format(scale)] = mean_mask.float()

outputs["var_mask/{}".format(scale)] = var_mask.float()

outputs["final_selection/{}".format(scale)] = final_mask.float()

```