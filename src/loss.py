from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F


class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean') -> Tensor:
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class HardNegativeContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, z: Tensor = None, reduction: str = 'mean') -> Tensor:
        # x: query embeddings
        # y: positive embeddings
        # z: negative embeddings (optional)
        
        if z is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, 
                device=x.device, dtype=torch.long)
            logits = torch.matmul(x, y.transpose(0, 1))
            loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
            return loss
            
        pos_logits = torch.matmul(x, y.transpose(0, 1)) 
        neg_logits = torch.matmul(x, z.transpose(0, 1)) 
        logits = torch.cat([pos_logits, neg_logits], dim=1) 
        
        target = torch.arange(x.size(0), device=x.device)

        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedHardNegativeContrastiveLoss(HardNegativeContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature=temperature)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, z: Tensor = None, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_z = self.gather_tensor(z) if z is not None else None
        
        loss = super().__call__(dist_x, dist_y, dist_z, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        if t is None:
            return None
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
