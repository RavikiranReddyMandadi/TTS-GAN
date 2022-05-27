import logging
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

class CausalConv1d(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        bias=True,
        pad="ConstantPad1d",
        pad_params={"value": 0.0},
    ):
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    def forward(self, x):

        return self.conv(self.pad(x))[:, :, : x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
        pad="ReplicationPad1d",
        pad_params={},
    ):
        super(CausalConvTranspose1d, self).__init__()
        # NOTE (yoneyama): This padding is to match the number of inputs
        #   used to calculate the first output sample with the others.
        self.pad = getattr(torch.nn, pad)((1, 0), **pad_params)
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )
        self.stride = stride

    def forward(self, x):
 
        return self.deconv(self.pad(x))[:, :, self.stride : -self.stride]


class ResidualBlock(torch.nn.Module):

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_causal_conv=False,
    ):

        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        self.use_causal_conv = use_causal_conv
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            if not use_causal_conv:
                self.convs1 += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=dilation,
                            bias=bias,
                            padding=(kernel_size - 1) // 2 * dilation,
                        ),
                    )
                ]
            else:
                self.convs1 += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        CausalConv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=dilation,
                            bias=bias,
                        ),
                    )
                ]
            if use_additional_convs:
                if not use_causal_conv:
                    self.convs2 += [
                        torch.nn.Sequential(
                            getattr(torch.nn, nonlinear_activation)(
                                **nonlinear_activation_params
                            ),
                            torch.nn.Conv1d(
                                channels,
                                channels,
                                kernel_size,
                                dilation=1,
                                bias=bias,
                                padding=(kernel_size - 1) // 2,
                            ),
                        )
                    ]
                else:
                    self.convs2 += [
                        torch.nn.Sequential(
                            getattr(torch.nn, nonlinear_activation)(
                                **nonlinear_activation_params
                            ),
                            CausalConv1d(
                                channels,
                                channels,
                                kernel_size,
                                dilation=1,
                                bias=bias,
                            ),
                        ),
                    ]

    def forward(self, x):

        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x

class Generator(torch.nn.Module):

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_causal_conv=False,
        use_weight_norm=True,
    ):

        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.use_causal_conv = use_causal_conv
        if not use_causal_conv:
            self.input_conv = torch.nn.Conv1d(
                in_channels,
                channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.input_conv = CausalConv1d(
                in_channels,
                channels,
                kernel_size,
                bias=bias,
            )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            if not use_causal_conv:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.ConvTranspose1d(
                            channels // (2 ** i),
                            channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                ]
            else:
                self.upsamples += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        CausalConvTranspose1d(
                            channels // (2 ** i),
                            channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            bias=bias,
                        ),
                    )
                ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]
        if not use_causal_conv:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                CausalConv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                ),
                torch.nn.Tanh(),
            )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class PeriodDiscriminator(torch.nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):

        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):
        fmap = []

        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        # outs = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
            # outs += [x]
        x = self.output_conv(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        # outs += [x]

        return x, fmap

    def apply_weight_norm(self):

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):

        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [PeriodDiscriminator(**params)]

    def forward(self, y, y_hat):

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(torch.nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[15, 41, 5, 3],
        channels=128,
        max_downsample_channels=1024,
        max_groups=16,
        bias=True,
        downsample_scales=[2, 2, 4, 4, 1],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):

        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels,
                    channels,
                    # NOTE(kan-bayashi): Use always the same kernel size
                    kernel_sizes[0],
                    bias=bias,
                    padding=(kernel_sizes[0] - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias=bias,
            ),
        ]

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):

        fmap = []

        for f in self.layers:
            x = f(x)
            fmap.append(x)

        x = torch.flatten(x, 1, -1)

        return x, fmap

    def apply_weight_norm(self):

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(
        self,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=False,
    ):

        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [ScaleDiscriminator(**params)]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

    def forward(self, y, y_hat):

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.pooling(y)
                y_hat = self.pooling(y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs