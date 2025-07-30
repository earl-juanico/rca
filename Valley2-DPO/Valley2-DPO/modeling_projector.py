import math
import torch
import torch.nn as nn

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'conv_adapter':
        return ConvAdapter(config.mm_hidden_size, config.hidden_size, getattr(config, "mlp_hidden_dim", None))
    elif projector_type == 'mlp_pixel_shuffle':
        return MlpPixelShuffle(config.mm_hidden_size, config.hidden_size,
                               config.pixelshuffle_downsample_ratio, getattr(config, "mlp_hidden_dim", None))
    elif projector_type == 'ovis_conv_adapter':
        return OvisConvAdapter(config.mm_hidden_size, config.hidden_size, getattr(config, "mlp_hidden_dim", 32000),
                               getattr(config, "tokenize_function", "softmax"))
    raise ValueError(f'Unknown projector type: {projector_type}')


class ConvAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, mlp_hidden_dim=None):
        super().__init__()
        self.mm_projector_type = 'conv_adapter'
        if mlp_hidden_dim is None:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.Linear(dim_out, dim_out)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim_out)
            )
        self.conv = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = self.mlp(x)

        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d).permute([0, 3, 1, 2])
        x = self.conv(x)
        x = x.permute([0, 2, 3, 1]).reshape(f, -1, d)
        return x


class MlpPixelShuffle(nn.Module):
    def __init__(self, dim_in, dim_out, pixelshuffle_downsample_ratio, mlp_hidden_dim=None):
        super().__init__()
        self.mm_projector_type = 'mlp_pixel_shuffle'
        if mlp_hidden_dim is None:
            self.mlp = nn.Sequential(
                nn.Linear(int(dim_in * (pixelshuffle_downsample_ratio ** 2)), dim_out),
                nn.GELU(),
                nn.Linear(dim_out, dim_out)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(int(dim_in * (pixelshuffle_downsample_ratio ** 2)), mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim_out)
            )
        self.scale_factor = pixelshuffle_downsample_ratio

    def pixel_shuffle(self, x, scale_factor=2):
        # change scale_factor from float to int

        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H / scale, C * scale
        x = x.view(n, w, int(h / scale_factor), int(c * scale_factor))
        # N, W, H / scale, C * scale --> N, H / scale, W, C * scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H / scale, W, C * scale --> N, H / scale, W / scale, C * (scale ** 2)
        x = x.view(n, int(h / scale_factor), int(w / scale_factor),
                   int(c * (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = x[:, 1:, :]  # remove cls_token
        h = w = int(x.shape[1] ** 0.5)
        x = x.view(x.shape[0], h, w, -1)
        x = self.pixel_shuffle(x, self.scale_factor)
        x = self.mlp(x)
        x = x.view(x.shape[0],-1,x.shape[-1])
        return x


class OvisConvAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, vocab_size, tokenize_function="softmax"):
        super().__init__()
        self.mm_projector_type = 'ovis_conv_adapter'
        self.conv = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_in, vocab_size, bias=False),
            torch.nn.LayerNorm(vocab_size)
        )
        self.embedding = torch.nn.Embedding(vocab_size, dim_out)
        self.tokenize_function = tokenize_function

    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.tokenize_function == 'softmax':
            tokens = torch.nn.functional.softmax(logits, dim=-1)
        elif self.tokenize_function == 'gumbel_argmax':
            tokens = torch.nn.functional.gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax,'
                f' but got {self.config.tokenize_function}'
            )
        return tokens

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        # conv
        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d).permute([0, 3, 1, 2])
        x = self.conv(x)
        x = x.permute([0, 2, 3, 1]).reshape(f, -1, d)

        # tokenize
        logits = self.mlp(x)
        visual_tokens = self.tokenize(logits)

        # get embeddings
        out = torch.matmul(visual_tokens, self.embedding.weight)

        return out
