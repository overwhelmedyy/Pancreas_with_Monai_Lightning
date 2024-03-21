from monai.networks.nets import UNETR, SwinUNETR, UNet, attentionunet, AttentionUnet, BasicUNetPlusPlus
from DUXNet.networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from DUXNet.networks.nnFormer.nnFormer_seg import nnFormer
from DUXNet.networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS



def create_network(network):
    if network == '3DUXNET':
        model = UXNET(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )
    elif network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=False,
        )
    elif network == 'nnFormer':
        model = nnFormer(input_channels=1, num_classes=2)
    elif network == 'UNETR':
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    # elif network == 'TransBTS':
    #     _, model = TransBTS(dataset=dataset, _conv_repr=True, _pe_type='learned')
    #     model = model
    elif network == "UNet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )
    elif network == "attentionunet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
    elif network == "UNetpp":
        model = BasicUNetPlusPlus(
            spatial_dims=3,
            features=(32, 32, 64, 128, 256, 32)
        )
    return model

def compute_network_parameter_number(network):
    return sum(p.numel() for p in network.parameters())

if __name__ == "__main__":
    network = "attentionunet"
    module = create_network(network)
    print(module)




# 3DUXNet: 53005922 53.01M
# SwinUNETR: 62186708 62.19M
# UNet: 4808917 4.81M
# nnFormer: 149247194 149.25M
# AttentionUnet: 5909130 5.91M
# UNet++: 6979976 6.98M
