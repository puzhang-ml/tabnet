import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GBN(nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)
        
    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)

class TabNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None
    ):
        """
        Defines TabNet encoder architecture.
        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
        n_d : int
            Dimension of the prediction layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of shared GLU layer in each GLU block (default 2)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : np.array
            Matrix of group relationships for features (optional)
        """
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)
        
        if group_attention_matrix is not None:
            self.group_attention_matrix = torch.from_numpy(group_attention_matrix.astype(np.float32))
        else:
            self.group_attention_matrix = None

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for step in range(n_steps):
            encoder = FeatTransformer(
                input_dim,
                n_d + n_a,
                n_independent,
                n_shared,
                virtual_batch_size,
                momentum
            )
            decoder = FeatTransformer(
                input_dim,
                n_d + n_a,
                n_independent,
                n_shared,
                virtual_batch_size,
                momentum
            )
            self.encoder.append(encoder)
            self.decoder.append(decoder)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)
        
        if prior is None:
            prior = torch.ones(x.shape).to(x.device)
            
        M_loss = 0
        att = torch.ones(x.shape[0], self.input_dim).to(x.device)
        steps_output = []
        
        for step in range(self.n_steps):
            M = self._compute_mask(prior, att, self.group_attention_matrix)
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M + 1e-10)), dim=1))
            
            masked_x = torch.mul(M, x)
            encoder_output = self.encoder[step](masked_x)
            decoder_output = self.decoder[step](masked_x)
            
            d = torch.sigmoid(decoder_output[:, :self.n_d])
            steps_output.append(d)
            
            # Update attention
            if step < self.n_steps - 1:
                att = torch.sigmoid(encoder_output[:, self.n_d:])
                att = self.gamma * att * (1 - M)
                
        M_loss /= self.n_steps
        return steps_output, M_loss

    def _compute_mask(self, prior, att, group_matrix=None):
        """Compute mask based on attention and group relationships"""
        mask = torch.mul(prior, att)
        
        if group_matrix is not None and self.group_attention_matrix is not None:
            group_att = torch.mm(mask, self.group_attention_matrix.to(mask.device))
            mask = torch.mm(group_att, self.group_attention_matrix.t().to(mask.device))
            
        if self.mask_type == "sparsemax":
            mask = sparsemax(mask, dim=-1)
        else:
            mask = entmax15(mask, dim=-1)
        return mask

class FeatTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_independent,
        n_shared,
        virtual_batch_size=128,
        momentum=0.02
    ):
        super(FeatTransformer, self).__init__()
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.shared = nn.ModuleList()
        self.independent = nn.ModuleList()
        
        # Shared layers
        for i in range(self.n_shared):
            if i == 0:
                self.shared.append(
                    GLU_Block(input_dim, output_dim, virtual_batch_size, momentum)
                )
            else:
                self.shared.append(
                    GLU_Block(output_dim, output_dim, virtual_batch_size, momentum)
                )

        # Independent layers
        for i in range(self.n_independent):
            if i == 0:
                self.independent.append(
                    GLU_Block(input_dim, output_dim, virtual_batch_size, momentum)
                )
            else:
                self.independent.append(
                    GLU_Block(output_dim, output_dim, virtual_batch_size, momentum)
                )

    def forward(self, x):
        # Independent GLU Blocks
        out = x
        for layer in self.independent:
            out = layer(out)
            
        # Shared GLU Blocks
        for layer in self.shared:
            out = layer(out)
        return out

class GLU_Block(nn.Module):
    """
    Gated Linear Unit block
    """
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        self.bn = GBN(2 * output_dim, virtual_batch_size, momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = F.glu(x, dim=1)
        return out

def sparsemax(z, dim=-1):
    """Sparsemax activation function"""
    z = z - torch.max(z, dim=dim, keepdim=True)[0]
    zs = torch.sort(z, dim=dim, descending=True)[0]
    range_idx = torch.arange(1, z.size(dim) + 1, device=z.device).float()
    bound = 1 + range_idx * zs
    cumsum_zs = torch.cumsum(zs, dim=dim)
    is_gt = bound > cumsum_zs
    k = torch.max(is_gt * range_idx, dim=dim, keepdim=True)[0]
    threshold = (cumsum_zs[torch.arange(z.size(0)), (k - 1).squeeze()] - 1) / k.squeeze()
    return torch.clamp(z - threshold.unsqueeze(dim), min=0)

def entmax15(z, dim=-1):
    """Entmax 1.5 activation function"""
    q = 1.5
    q_minus_1 = 0.5
    
    z = z - torch.max(z, dim=dim, keepdim=True)[0]
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    
    z_sum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device).float()
    w = z_sorted - (z_sum - 1) / k
    
    threshold = w[torch.arange(z.size(0)), (torch.sum(w > 0, dim=dim) - 1)]
    return torch.clamp(torch.pow(F.relu(z - threshold.unsqueeze(dim)), q_minus_1), min=0) 