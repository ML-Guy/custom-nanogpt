import inspect
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import List

from model import *


class CodebookDecoder(nn.Module):
    def __init__(self, num_elements: int = 1000, embedding_dim: int = 256, num_blocks: int = 3, ema_decay=0,
                 resampling: str = "noise", temperature:float = 1, capacity_factor:float =1.5 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_elements = num_elements
        self.embedding_dim = embedding_dim
        
        self.embedding_table = nn.ModuleList([nn.Embedding(num_elements, embedding_dim) for _ in range(num_blocks)])

        self.resampling_method = resampling
        self.temperature = temperature
        self.dist_temp = nn.Parameter(torch.ones(1))
        self.capacity_factor = capacity_factor

        self.embedding_decoder =  nn.ModuleList([nn.Linear(embedding_dim, num_elements, bias=False) for _ in range(num_blocks)])
        for i in range(num_blocks):
            self.embedding_table[i].weight = self.embedding_decoder[i].weight


        self.ema_decay = ema_decay
        if self.ema_decay:
            # self.ema_embedding_table = self.embedding_table.clone().detach() # Dont exist
            self.ema_embedding_table = nn.ModuleList([nn.Embedding(num_elements, embedding_dim) for _ in range(num_blocks)])
            for i in range(self.num_blocks):
                self.ema_embedding_table[i].weight.data.copy_(self.embedding_table[i].weight.data)    


    def forward(self, x: torch.Tensor) -> List:
        # Reshape ouput: (B, T , num_blocks * embedding_dim) -> (B, T, num_blocks, embedding_dim)
        B, T, embd_dim = x.size()
        x = x.view(B, T, self.num_blocks, self.embedding_dim)

        decoded_indices = []
        decoded_latents = []
        dist_logits = []

        # clamp distance temperature between 0.001, 10
        # temperature = torch.clamp(temperature, min=0.001, max=10.0) # it doesn't allow grad update once value is out of bound
        if self.dist_temp >10.0 or self.dist_temp < 0.001:
            self.dist_temp.data.copy_(10.0 if self.dist_temp >1.0 else 0.001)
        
        # Calculate expert capacity
        tokens_per_batch = B * T
        expert_capacity = int((tokens_per_batch / self.num_elements) * self.capacity_factor)

        for i in range(self.num_blocks):
            embedding = x[:,:,i,:]
 
            dist_logit = self.embedding_decoder[i](embedding) # B, T, num_elements
            
            # Apply noise resampling with balanced token distribution
            noise = 1 - self.temperature * torch.rand_like(dist_logit)
            noisy_dist = dist_logit * noise
            
            # Use topk to select the top expert_capacity elements for each expert
            topk_values, topk_indices = torch.topk(noisy_dist, k=expert_capacity, dim=1)
            
            # Create a mask for the selected indices
            mask = torch.zeros_like(noisy_dist, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            
            # Assign tokens to experts, argmax to select the expert with the highest noisy probability
            embedding_index = torch.argmax(mask.float() * noisy_dist, dim=-1)

            decoded_latent = self.embedding_table[i](embedding_index)

            decoded_indices.append(embedding_index)
            decoded_latents.append(decoded_latent)
            dist_logits.append(dist_logit)

        decoded_indices = torch.stack(decoded_indices, dim=2)  # B, T, num_block
        decoded_latents = torch.cat(decoded_latents, dim=2)  # B, T, num_block * embd 
        dist_logits = torch.stack(dist_logits, dim=2)  # B, T, num_block, num_elements

        return decoded_indices, decoded_latents, dist_logits
        
    def decode(self, indices: torch.Tensor ) -> torch.Tensor:
        embeddings = [self.embedding_table[i](indices[..., i]) for i in range(self.num_blocks)]
        return torch.cat(embeddings, dim=2) # B, T, Embeddings
    
    def ema_update(self):
        if self.ema_decay:
            with torch.no_grad():
                for i in range(self.num_blocks):
                    self.ema_embedding_table[i].weight.data = self.ema_decay * self.ema_embedding_table[i].weight.data + (1 - self.ema_decay) * self.embedding_table[i].weight.data
                    self.embedding_table[i].weight.copy_(self.ema_embedding_table[i].weight)

# @torch.compile
def get_vq_loss(tok_emb, vq_indices, vq_tok_emb, vq_in_logits, 
            out_emb, vq_out_indices, vq_out_embd, vq_out_logits,
            target_emb, vq_tgt_indices, vq_tgt_embd, vq_tgt_logits, 
            num_blocks, vq_beta: dict[str,float])-> dict:
    
    loss = {}
    num_elements = vq_tgt_logits.size(-1)

    # Commitment loss: Input
    # vq_commit_loss = F.mse_loss(vq_tok_emb.detach(), tok_emb)
    # vq_embd_loss =  F.mse_loss(vq_tok_emb, tok_emb.detach())
    vq_in_ce_loss = F.cross_entropy(vq_in_logits.view(-1, num_elements), vq_indices.view(-1))

    # Target loss
    # assumption target indicies are correct -> source logit should point towards target indices 
    # Below CE loss Updates model logits and codebook symmetricaly
    vq_out_ce_loss = F.cross_entropy(vq_out_logits.view(-1, num_elements), vq_tgt_indices.view(-1))

    # assumption decoded indicies are correct -> target logit should point towards decoded indices 
    # Below CE loss Updates token embedding and codebook symmetricaly
    vq_tgt_ce_loss = F.cross_entropy(vq_tgt_logits.view(-1, num_elements), vq_out_indices.view(-1))
    vq_loss = vq_beta["vq_out_ce_loss"] * vq_out_ce_loss \
            + vq_beta["vq_tgt_ce_loss"] * vq_tgt_ce_loss \
            + vq_beta["vq_embd_loss"] * vq_in_ce_loss
            # + vq_beta["vq_commit_loss"] * vq_commit_loss \

    loss["vq_loss"] = vq_loss
    # loss["vq_commit_loss"] = vq_commit_loss
    loss["vq_in_ce_loss"] = vq_in_ce_loss
    loss["vq_tgt_ce_loss"] = vq_tgt_ce_loss
    loss["vq_out_ce_loss"] = vq_out_ce_loss

    return loss



@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 1 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    cb_num_elements = 256
    cb_embedding_dim = 128
    cb_num_blocks = 6
    cb_ema_decay = 0
    cb_resampling_temp = 0.02
    vq_beta = {
        "vq_out_ce_loss": 1.0,
        "vq_tgt_ce_loss": 1.0,
        "vq_commit_loss": 0.25,
        "vq_embd_loss": 1.0,
        "vq": 0.3
    }


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.cb_decoder = CodebookDecoder(
                num_elements=config.cb_num_elements, embedding_dim=config.cb_embedding_dim,
                num_blocks= config.cb_num_blocks, ema_decay=config.cb_ema_decay,
                temperature=config.cb_resampling_temp
                )

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)

        # VQ Codebook coding
        vq_indices, vq_tok_emb, vq_in_logits = self.cb_decoder(tok_emb)
        vq_tok_emb_reparam = tok_emb + (vq_tok_emb-tok_emb).detach()
          
        x = vq_tok_emb_reparam + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        # VQ Logits and indicies
        vq_out_indices, vq_out_embd, vq_out_logits = self.cb_decoder(x)
        vq_out_reparam = x + (vq_out_embd-x).detach()


        # Token decoding logits
        logits = self.lm_head(vq_out_reparam) # (B, T, vocab_size)

        loss = {}
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss["ce_loss"] = ce_loss

            # target embeddings 
            target_emb = self.transformer.wte(targets) # B, T, n_embd
            vq_tgt_indices, vq_tgt_embd, vq_tgt_logits = self.cb_decoder(target_emb)

            
            # Cross Entropy loss  - correct target # bring logits near correct target logits

            vq_loss = get_vq_loss(tok_emb, vq_indices, vq_tok_emb, vq_in_logits, 
                                  x, vq_out_indices, vq_out_embd, vq_out_logits,
                                  target_emb, vq_tgt_indices, vq_tgt_embd, vq_tgt_logits, 
                                  self.config.cb_num_blocks, self.config.vq_beta)
            
            loss.update(vq_loss)
            loss["loss"] = ce_loss + self.config.vq_beta["vq"] * vq_loss["vq_loss"]
            
        return logits, vq_out_indices, loss

    @classmethod
    def from_pretrained(cls, model_type):
        raise NotImplementedError
        
    def configure_optimizers(self, weight_decay, learning_rate, device_type, print_log=True):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if print_log:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if print_log:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
