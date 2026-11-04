"""Fold a learned CL-Last readout into the existing Packet affine projection."""
import copy
import torch


def fold_last(projection,fusion):
    if projection.weight.shape!=(64,128) or fusion.packet_projection.weight.shape!=(32,64):
        raise ValueError('Expected registered CL-Last128/64 and Packet64/32 projections')
    result=copy.deepcopy(fusion)
    with torch.no_grad():
        # Compose in float64, then round once to the source parameter dtype.
        # Inference remains float32; exact output bits are not assumed equal.
        w=fusion.packet_projection.weight.double()
        q=projection.weight.double()
        result.packet_projection.weight.copy_(w@(q[:,:64]+q[:,64:]))
        result.packet_projection.bias.copy_(w@projection.bias.double()+fusion.packet_projection.bias.double())
    return result.eval()
