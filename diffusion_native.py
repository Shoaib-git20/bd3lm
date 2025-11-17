import itertools
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from torch.distributed.tensor import DTensor, Replicate, Shard

import models
import noise_schedule
import utils

def _sample_categorical(categorical_probs):
  """Sample from a categorical distribution."""
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  """Unsqueeze x to match the number of dimensions of reference."""
  return x.view(
    *x.shape,
    *((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor

class Diffusion(torch.nn.Module):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale

    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id

    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization

    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length

    if self.parameterization == 'ar':
      self.block_size = 1

    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.sampling_eps_min = torch.tensor(self.config.training.sampling_eps_min)
      self.sampling_eps_max = torch.tensor(self.config.training.sampling_eps_max)
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self._validate_configuration()

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(), self.noise.parameters())

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {'dit', 'ar', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    device = next(self.parameters()).device
    if hasattr(self.backbone, "block_diff_mask"):
        if self.config.model.attn_backend == 'sdpa':
            self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(device)
        elif self.config.model.attn_backend == 'flex':
            self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
        self.sampling_eps_min = self.sampling_eps_min.to(device)
        self.sampling_eps_max = self.sampling_eps_max.to(device)
    return self

  def _replace_ckpt_keys(self, state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    return new_state_dict

  def load_state_dict(self, state_dict, strict=True):
    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(state_dict.keys())[0]:
      state_dict = self._replace_ckpt_keys(state_dict)
    
    # Separate EMA state dict if it exists
    ema_state_dict = None
    if 'ema' in state_dict:
        ema_state_dict = state_dict.pop('ema')

    # Load model state dict
    model_state_dict = {k: v for k, v in state_dict.items() if k not in ['ema', 'sampling_eps_min', 'sampling_eps_max']}
    
    # Handle missing keys if not strict
    if not strict:
        new_model_state_dict = self.state_dict()
        new_model_state_dict.update(model_state_dict)
        model_state_dict = new_model_state_dict

    super().load_state_dict(model_state_dict, strict=strict)

    if self.ema and ema_state_dict:
      self.ema.load_state_dict(ema_state_dict)
    if 'sampling_eps_min' in state_dict:
      self.sampling_eps_min = state_dict['sampling_eps_min']
      self.sampling_eps_max = state_dict['sampling_eps_max']

  def state_dict(self, *args, **kwargs):
    state_dict = super().state_dict(*args, **kwargs)
    if self.ema:
      state_dict['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      state_dict['sampling_eps_min'] = self.sampling_eps_min
      state_dict['sampling_eps_max'] = self.sampling_eps_max
    return state_dict

  def _subs_parameterization(self, logits, xt):
    logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
    logits = torch.scatter(logits, -1, xt[..., None], torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name == 'bd3lm':
        logits = self.backbone(x, sigma, store_kv=store_kv, sample_mode=sample_mode)
      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn:
      x = x[:, :self.config.model.length]
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits, xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
    return logits
    
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds."""
    # Check if we are in Distributed Mode
    is_dtensor = isinstance(x, DTensor)

    # 1. Work with Local Tensors to avoid DTensor indexing issues
    if is_dtensor:
        local_xt = xt.to_local()
        local_x = x.to_local()
        local_move_indices = move_indices.to_local()
        local_p = p.to_local()
    else:
        local_xt = xt
        local_x = x
        local_move_indices = move_indices
        local_p = p

    # 2. Perform Logic on Local Tensors
    mask_val = self.mask_index
    if isinstance(mask_val, torch.Tensor):
        mask_val = mask_val.item()
        
    perc_masked = (local_xt == mask_val).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or (perc_masked > sampling_eps_max).any():
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0: break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0: break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      else: # No resampling needed for the default full range
          break
      
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      local_move_indices[regen_idx] = (torch.rand(*local_x.shape, device=local_x.device) < local_p)[regen_idx]
      local_xt = torch.where(local_move_indices, mask_val, local_x)
      local_xt = local_xt.reshape(local_xt.shape[0], -1, block_size)
      perc_masked = (local_xt == mask_val).float().sum(-1) / block_size

    # 3. Re-Wrap results into DTensor if necessary
    if is_dtensor:
      local_xt = local_xt.reshape(local_xt.shape[0], -1) # Flatten back to match original shape
      # Create new DTensors with updated data, We use the mesh/placements from the original 'x'
      xt = DTensor.from_local(local_xt, x.device_mesh, x.placements)
    else:
      xt = local_xt.reshape(local_xt.shape[0], -1)
    return xt
  
  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt."""
    if block_size is None:
      block_size = self.block_size
    if isinstance(x, DTensor):
        local_x = x.to_local()
        local_p = p.to_local()
        move_indices_local = torch.rand(*local_x.shape, device=local_x.device) <= local_p
        
        move_indices = DTensor.from_local(
            move_indices_local,
            x.device_mesh,
            x.placements
        )
    else:
        move_indices = torch.rand(*x.shape, device=x.device) <= p

    mask_val = self.mask_index
    if isinstance(mask_val, torch.Tensor):
        mask_val = mask_val.item()

    xt = torch.where(move_indices, mask_val, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    if self.config.training.resample and not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64, device=next(self.parameters()).device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if self.config.sampling.kv_cache:
        p_x0 = self.forward(x[:, -self.block_size:], sigma_t, sample_mode=True).to(torch.float64)
      else:   
        p_x0 = self.forward(x, sigma_t, sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new

  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    device = next(self.parameters()).device
    with torch.amp.autocast('cuda', dtype=torch.float32):
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros((bsz, num_pred_tokens + 1), dtype=torch.long, device=device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        noise = (torch.distributions.Gumbel(0, 1).sample((bsz, self.vocab_size))).to(device)
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def sample(self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    self.eval()
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
      
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    
    samples = []
    total_nfes = []

    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(batch_size_per_gpu)
          if num_tries > 10: raise ValueError('Sampling failed.')
        samples.append(sample_i)
        total_nfes.append(self.config.model.length)
    elif self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=(seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10: raise ValueError('Sampling failed.')
        samples.append(sample_i)
        total_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          num_tries += 1
          if num_tries > 10 and sample_i is None: raise ValueError('Sampling failed.')
        samples.append(sample_i)
        total_nfes.append(nfes)
    
    if self.ema:
        self.ema.restore(self._get_parameters())

    samples = torch.cat(samples, dim=0) 
    return self.tokenizer.batch_decode(samples), total_nfes

  def _sigma_from_p(self, p):
    return torch.min(-torch.log(1 - p), self.noise.sigma_max)

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return model_output.exp()
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[..., None]
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if t is None:
      t = self._sample_t(x0.shape, x0.device, sampling_eps_min, sampling_eps_max)

    loss_scale, p = self.noise(t)

    dp_rank, local_bs = None, None
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma)

    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    # --- sigma and p convert to DTensor ---
    if isinstance(x0, DTensor):
        local_x = x0.to_local()
        local_sigma = sigma
        # --- Convert 'sigma' ---
        if sigma.shape[0] != local_x.shape[0]:
            dp_rank = x0.device_mesh.get_local_rank(mesh_dim=0)
            local_bs = local_x.shape[0]
            start, end = dp_rank * local_bs, (dp_rank + 1) * local_bs
            local_sigma = sigma[start:end]

        local_sigma = local_sigma.to(x0.device_mesh.device_type)
        
        sigma = DTensor.from_local(
            local_sigma, x0.device_mesh, [Shard(0), Replicate()]
        )
        # --- Convert 'p' ---
        local_p = p
        if p.shape[0] != local_x.shape[0]:
            local_p = p[start:end]
            
        local_p = local_p.to(x0.device_mesh.device_type)
        
        p = DTensor.from_local(
            local_p, x0.device_mesh, [Shard(0), Replicate()]
        )

    xt = self.q_xt(x0, p, sampling_eps_min=sampling_eps_min, sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(model_output, sigma, xt, x0)

    log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

  def compute_loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min.item()
      sampling_eps_max = self.sampling_eps_max.item()
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
      
    (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(x0, attention_mask)
    
    if self.parameterization == 'ar':
      output = self.forward(input_tokens, None)
      loss = - output.gather(-1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(
        input_tokens,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max)
    
    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)

  def update_clipped_schedule(self, valid_vars):
    best_var = float('inf')
    sampling_eps_min_best, sampling_eps_max_best = self.sampling_eps_min, self.sampling_eps_max
    
    for (eps_min, eps_max), var_list in valid_vars.items():
      if not var_list: continue
      all_vars = torch.cat(var_list).var()
      
      print(f'Variance for interval [{eps_min:.2f} - {eps_max:.2f}]: {all_vars.item()}')

      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max

    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)
      print(f"Updated sampling schedule to eps_min={self.sampling_eps_min.item()}, eps_max={self.sampling_eps_max.item()}")


  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss."""
    masked_indices = xt == self.mask_index
    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]
    words_that_were_masked = x0[masked_indices]
    neg_term = q_ratio * torch.gather(log_score[masked_indices], -1, words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)
    entropy = torch.zeros(*xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def _analytic_sampler(self, n_samples, num_steps, seqlen, eps=1e-5): 
    device = next(self.parameters()).device
    x = self._sample_prior(n_samples, seqlen).to(device)
    x[:, 0] = self.tokenizer.bos_token_id
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps
    for i in tqdm(range(num_steps), desc='step'):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
      x = self._analytic_update(x=x, t=t, dt=dt)
    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    x = self._denoiser_update(x=x, t=t)
    
    stop, x = self._check_stop_conds(x)
    if stop:
      return None
    return x

  @torch.no_grad
  def _semi_ar_sampler(self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
    device = next(self.parameters()).device
          
    mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
    if mdlm_semi_ar:
      num_strides = self.config.model.length // 512 - 1

    ones = torch.ones((n_samples,1), dtype=next(self.parameters()).dtype, device=device)
    
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(device)
        x_accum[:, 0] = self.tokenizer.bos_token_id
      else:
        x = self._sample_prior(n_samples, 512 if mdlm_semi_ar else self.block_size).to(device)
        x_accum = torch.cat((x_accum, x), dim=1)

      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)
      if mdlm_semi_ar and stride_num > 0:
        fwd_idx = torch.arange(512*stride_num, (512*stride_num)+self.block_size)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          if num_masked > 0:
            t *= u**(1 / num_masked)
        else:
          t = timesteps[i]

        p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx], t=t * ones, dt=dt, p_x0=p_x0_cache)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if (stop and not self.config.sampling.var_length) or (stop and x.shape[-1] == 1):
          return None, None
        elif stop:
          break
    return x_accum, sampling_steps
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    stop = False
    truncate_idx = None

    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    if self.config.sampling.var_length:
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
