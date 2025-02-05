from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as nps

from decoder.layer_library import GeneratorUnetFromLatent, EConvLayer, E_Normalization_Type, Discriminator, EActivationType


import pytorch_lightning as pl

import random
from sklearn.decomposition import NMF





class ReLU_AllowGradFromNegative(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output



class FeatureExtractorForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.identity_layer_for_output_hook = nn.Identity()

    def forward(self, x):
        #print("x.shape: {}".format(x.shape))
        return self.identity_layer_for_output_hook(self.model(x)['out'])



class Decoder(pl.LightningModule):
    def __init__(self, feature_extractor, default_noise_size=224,
                 normalization=torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 **kwargs):

        super().__init__(**kwargs)
        torch.manual_seed(42)
        self.noise = torch.randn(1, 1, default_noise_size, default_noise_size)
        self.set_models(feature_extractor=feature_extractor, default_noise_size=default_noise_size, normalization=normalization)
        activations = feature_extractor(torch.zeros(1, 3, 224, 224))['out']
        if len(activations.shape) == 4:
            self.num_channels = activations.shape[1]
        elif len(activations.shape) == 3:
            self.num_channels = activations.shape[2]


    def set_models(self, feature_extractor, default_noise_size=224,
                 normalization=torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
        #super().__init__(**kwargs)
        self.feature_extractor = feature_extractor#.to("cuda")
        activations = self.feature_extractor(torch.zeros(1, 3, 224, 224))['out']
        if len(activations.shape) == 4:
            num_channels = activations.shape[1]
        elif len(activations.shape) == 3:
            num_channels = activations.shape[2]
        else:
            raise RuntimeError

        self.decoder = GeneratorUnetFromLatent(kernel_size_full_res_conv=None, kernel_size_encoder=4,
                    kernel_size_decoder=4, n_channels_input=1,
                    n_channels_out_encoder=[32, 64, 128, 256, 512, 512], n_channels_out_decoder=[512, 256, 128, 64, 32, 16], n_ch_out=3,
                    use_head_block=True,
                    n_resnet_blocks_encoder=1,
                    n_resnet_blocks_decoder=1,
                    bias_initialization=False,
                    e_conv_layer=EConvLayer.conv2d,
                    e_normalization_type_encoder=[E_Normalization_Type.no_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm,
                                                  E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm],
                    e_normalization_type_decoder=E_Normalization_Type.batch_norm,
                    n_channels_from_seg_net=num_channels,
                    apply_tanh=False)


        self.decoder.default_noise_size = default_noise_size

        self.normalization = normalization
        self.epoch = 0

        return num_channels


    def _run_decoder_from_activation_vectors(self, activation_vectors, noise=None):
        return self.decoder(activation_vectors, noise=noise)


    def forward(self, activation_vectors, fixed_noise=True):
        if fixed_noise:
            noise = torch.cat([self.noise for _ in range(activation_vectors.shape[0])], dim=0).to(activation_vectors.device)
        else:
            noise = None
        feature_visualizations = self._run_decoder_from_activation_vectors(activation_vectors, noise=noise)
        #feature_visualizations = self.decoder(activation_vectors)
        return F.sigmoid(feature_visualizations)


    def noise_forward(self, noise):
        return self.decoder(torch.zeros(noise.shape[0], 256).to("cuda"), noise=noise[:, 0].unsqueeze(dim=1))

    def forward_no_inputs(self, batch_size):
        #if self.noise is None:
        #    self.noise = torch.randn(batch_size, 1, self.decoder.default_noise_size, self.decoder.default_noise_size).to("cuda")
        #return F.sigmoid(self.decoder(torch.zeros(shape_like.shape[0], 128).to("cuda"), self.noise))

        noise = torch.cat([self.noise for _ in range(batch_size)], dim=0).to(self.device)
        dummy_activation_vectors = torch.randn((batch_size, self.num_channels), device=self.device)
        return F.sigmoid(self.decoder(dummy_activation_vectors, noise=noise))


    def set_image_samples(self, x):
        self.x = x


    def training_step(self, batch, idx):
        self.feature_extractor.eval()
        x = batch[0]
        with torch.no_grad():
            activations = self.feature_extractor(x)["out"]
            # cnn
            if len(activations.shape) == 4:
                idx_h = random.randint(0, activations.shape[2]-1)
                idx_w = random.randint(0, activations.shape[3]-1)
                activation_vectors = activations[:, :, idx_h, idx_w]
            # vit
            elif len(activations.shape) == 3:
                idx = random.randint(1, activations.shape[1]-1) # first entry is class token, skip that  # last entry is output embedding, or maybe first entry? Skip both
                activation_vectors = activations[:, idx]
            else:
                raise RuntimeError

        feature_visualizations = self._run_decoder_from_activation_vectors(activation_vectors)#self.decoder(activation_vectors)
        feature_visualizations = F.sigmoid(feature_visualizations)

        feature_visualizations = self.normalization(feature_visualizations)
        activation_feature_visualiziations = self.feature_extractor(feature_visualizations)["out"]

        #activation_vectors = F.relu(activation_vectors)
        #activation_feature_visualiziations = ReLU_AllowGradFromNegative.apply(activation_feature_visualiziations)
        if len(activations.shape) == 4:
            loss = F.l1_loss(torch.mean(activation_feature_visualiziations, dim=[2, 3]), activation_vectors)
        elif len(activations.shape) == 3:
            loss = F.l1_loss(torch.mean(activation_feature_visualiziations[:, 1:], dim=[1]), activation_vectors)   # ignore class token
        else:
            raise RuntimeError


        #mean_feature_vis = torch.mean(activation_feature_visualiziations, dim=[2, 3])
        #loss = -(torch.mean(mean_feature_vis)**0.5)*torch.mean(F.cosine_similarity(mean_feature_vis, activation_vectors))**2

        self.log("L1 Loss", loss)
        return loss


    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.eval()
        with torch.no_grad():
            activations = self.feature_extractor(self.x.to(self.device))["out"]
            if (len(activations.shape)==4):
                activation_vectors = activations[:, :, int(activations.shape[2]/2), int(activations.shape[3]/2)]
            else:
                activation_vectors = activations[:, 12]

            feature_visualizations = self.decoder(activation_vectors)
            feature_visualizations = F.sigmoid(feature_visualizations)
            torchvision.utils.save_image(feature_visualizations, "logs/images/imgs_" + str(self.epoch) + ".jpg")
            self.epoch += 1
            self.x.cpu()


        self.train()
        return super().training_epoch_end(outputs)


    def configure_optimizers(self):
        return [torch.optim.Adam(self.decoder.parameters())], []



class Decoder_LinearLayers(Decoder):
    def set_models(self, **kwargs):
        num_channels = super().set_models(**kwargs)

        self.linear_layers = nn.Sequential()
        self.linear_layers.add_module('linear_1', nn.Linear(num_channels, 512))
        self.linear_layers.add_module('relu_1', nn.ReLU(inplace=True))
        self.linear_layers.add_module('linear_2', nn.Linear(512, 512))
        self.linear_layers.add_module('relu_2', nn.ReLU(inplace=True))
        self.linear_layers.add_module('linear_3', nn.Linear(512, 512))
        self.linear_layers.add_module('relu_3', nn.ReLU(inplace=True))
        self.linear_layers.add_module('linear_4', nn.Linear(512, num_channels))

    def _run_decoder_from_activation_vectors(self, activation_vectors, noise=None):
        activation_vectors = self.linear_layers(activation_vectors)
        return self.decoder(activation_vectors, noise=noise)



class Decoder_Digipath(Decoder):

    def set_models(self, feature_extractor, default_noise_size=512,
                 normalization=torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
        self.feature_extractor = feature_extractor#.to("cuda")
        activations = self.feature_extractor(torch.zeros(1, 3, default_noise_size, default_noise_size))['out']
        if len(activations.shape) == 4:
            num_channels = activations.shape[1]
        elif len(activations.shape) == 3:
            num_channels = activations.shape[2]
        else:
            raise RuntimeError

        self.decoder = GeneratorUnetFromLatent(kernel_size_full_res_conv=None, kernel_size_encoder=4,
                    kernel_size_decoder=4, n_channels_input=1,
                    n_channels_out_encoder=[32, 64, 128, 256, 512, 512, 512], n_channels_out_decoder=[512, 512, 256, 128, 64, 32, 16], n_ch_out=3,
                    use_head_block=True,
                    n_resnet_blocks_encoder=1,
                    n_resnet_blocks_decoder=1,
                    bias_initialization=False,
                    e_conv_layer=EConvLayer.conv2d,
                    e_normalization_type_encoder=[E_Normalization_Type.no_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm,
                                                  E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm],
                    e_normalization_type_decoder=E_Normalization_Type.batch_norm,
                    n_channels_from_seg_net=num_channels,
                    apply_tanh=False)


        self.decoder.default_noise_size = default_noise_size

        self.normalization = normalization
        self.epoch = 0

        self.discriminator = Discriminator(kernel_size=4,
                            e_conv_layer=EConvLayer.bilinear_conv_first,
                            n_channels_out=[32, 64, 128, 256, 512, 512], n_channels_in=[3, 32, 64, 128, 256, 512],
                            n_resnet_blocks=0, e_normalization_type=[E_Normalization_Type.no_norm, E_Normalization_Type.instance_norm_running, E_Normalization_Type.instance_norm_running,
                                E_Normalization_Type.instance_norm_running, E_Normalization_Type.instance_norm_running, E_Normalization_Type.instance_norm_running],
                                use_spectral_norm=True,
                                e_activation_type=EActivationType.leaky_relu
                            )



    #def _run_decoder_from_activation_vectors(self, activation_vectors):
    #    feature_visualizations = self.decoder(activation_vectors)
    #    return F.interpolate(feature_visualizations, size=512, mode="bilinear")


    def get_activation_vectors(self, x):
        #print("x.shape: {}".format(x.shape))
        b_is_vit = False
        with torch.no_grad():
            activations = self.feature_extractor(x)["out"]
            # cnn
            if len(activations.shape) == 4:
                idx_h = random.randint(0, activations.shape[2]-1)
                idx_w = random.randint(0, activations.shape[3]-1)
                activation_vectors = activations[:, :, idx_h, idx_w]
            # vit
            elif len(activations.shape) == 3:
                idx = random.randint(1, activations.shape[1]-1) # first entry is class token, skip that  # last entry is output embedding, or maybe first entry? Skip both
                activation_vectors = activations[:, idx]
                b_is_vit = True
            else:
                raise RuntimeError

        return activation_vectors, b_is_vit


    def training_step(self, batch, idx, optimizer_idx):
        self.feature_extractor.eval()
        x = batch[0]
        activation_vectors, b_is_vit = self.get_activation_vectors(x)

        feature_visualizations = self._run_decoder_from_activation_vectors(activation_vectors)#self.decoder(activation_vectors)
        feature_visualizations = F.sigmoid(feature_visualizations)

        if optimizer_idx == 0:
            disc_out_fake = self.discriminator(feature_visualizations-0.5)

            size = disc_out_fake.shape[-1]
            #size = 64
            real = torch.ones((feature_visualizations.shape[0], 1, size, size)).to(feature_visualizations.device)
            fake = torch.zeros((feature_visualizations.shape[0], 1, size, size)).to(feature_visualizations.device)
            #print("disc out fake shape: {}".format(disc_out_fake.shape))
            disc_loss = self.adversarial_loss(disc_out_fake, real)


            feature_visualizations = self.normalization(feature_visualizations)
            activation_feature_visualiziations = self.feature_extractor(feature_visualizations)["out"]

            #activation_vectors = F.relu(activation_vectors)
            #activation_feature_visualiziations = ReLU_AllowGradFromNegative.apply(activation_feature_visualiziations)
            if (not b_is_vit):
                reconstruction_loss = F.l1_loss(torch.mean(activation_feature_visualiziations, dim=[2, 3]), activation_vectors)
            elif b_is_vit:
                reconstruction_loss = F.l1_loss(torch.mean(activation_feature_visualiziations[:, 1:], dim=[1]), activation_vectors)   # ignore class token
            else:
                raise RuntimeError

            loss = reconstruction_loss + disc_loss*0.1


            #mean_feature_vis = torch.mean(activation_feature_visualiziations, dim=[2, 3])
            #loss = -(torch.mean(mean_feature_vis)**0.5)*torch.mean(F.cosine_similarity(mean_feature_vis, activation_vectors))**2

            self.log("L1 Loss", reconstruction_loss)
            self.log("Gen Loss", disc_loss)


        elif optimizer_idx == 1:
            disc_out_fake = self.discriminator(feature_visualizations-0.5)
            disc_out_real = self.discriminator(x-0.5)

            size = disc_out_fake.shape[-1]
            #size = 64
            real = torch.ones((x.shape[0], 1, size, size)).to(x.device)
            fake = torch.zeros((feature_visualizations.shape[0], 1, size, size)).to(feature_visualizations.device)
            #print("disc out fake shape: {}".format(disc_out_fake.shape))
            loss = 0.5*self.adversarial_loss(disc_out_fake, fake) + 0.5*self.adversarial_loss(disc_out_real, 0.9*real)

            self.log("Disc Loss", loss)


        return loss


    def adversarial_loss(self, y_hat, y, label_smoothing=0.0):
        return F.binary_cross_entropy_with_logits(y_hat, y)


    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [opt_generator, opt_discriminator], []

        #return [torch.optim.Adam(self.decoder.parameters()), torch.optim.Adam(self.discriminator.parameters())], []






def nmf_decompose(input_activation_bchw, n_components=6):
        input_for_nmf = input_activation_bchw
        b, c, h, w = input_for_nmf.shape

        input_for_nmf = torch.reshape(input_for_nmf, shape=(input_for_nmf.shape[1], input_for_nmf.shape[0]*input_for_nmf.shape[2]*input_for_nmf.shape[3]))
        input_for_nmf = torch.permute(input_for_nmf, (1, 0))

        nmf = NMF(n_components=n_components)

        X_embedded = nmf.fit_transform(input_for_nmf)
        #basis = nmf.components_

        #reconstruct = torch.permute(torch.from_numpy(np.matmul(X_embedded, basis)), (1, 0))


        #reconstruct = torch.reshape(reconstruct, shape=(b, c, h, w))

        X_embedded = torch.from_numpy(X_embedded).float()
        X_embedded = torch.permute(X_embedded, (1, 0))
        #print("X_embedded.shape: {}".format(X_embedded.shape))
        X_embedded = torch.reshape(X_embedded, shape=(n_components, h, w))

        return X_embedded#torch.from_numpy(basis), X_embedded




class DecoderDigipath_NMF(Decoder_Digipath):

    def get_activation_vectors(self, x):
        #print("x.shape: {}".format(x.shape))
        b_is_vit = False
        activation_vectors = []
        with torch.no_grad():
            activations = self.feature_extractor(x)["out"]
            for activation in activations:
                X_embedded = nmf_decompose(activation.unsqueeze(dim=0).cpu()).to("cuda")

                for activation_channel in X_embedded:
                    activation_vec = torch.mean(
                        activation * activation_channel.unsqueeze(dim=0).unsqueeze(dim=0),
                        dim=[0, 2, 3],
                    ) / torch.mean(activation_channel)
                    activation_vectors.append(activation_vec)
            activation_vectors = torch.stack(activation_vectors, dim=0).to(self.device)


        return activation_vectors, b_is_vit


