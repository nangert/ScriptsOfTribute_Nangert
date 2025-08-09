import torch, torch.nn as nn, torch.nn.functional as F

class RNDHead(nn.Module):
    """Random Network Distillation on policy features z∈R^128 (detached)."""
    def __init__(self, feat_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        def mlp(in_d, out_d):
            return nn.Sequential(nn.Linear(in_d, 256), nn.ReLU(), nn.Linear(256, out_d))
        self.target = mlp(feat_dim, proj_dim)
        self.predictor = mlp(feat_dim, proj_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def target_embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.target(z.float())

    def predict_embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z.float())

    @torch.no_grad()
    def intrinsic_reward(self, z: torch.Tensor) -> torch.Tensor:
        """MSE(pred, target) over feature dim; z is expected detached."""
        t = self.target_embed(z)
        p = self.predict_embed(z)
        return ((p - t) ** 2).mean(dim=-1)

    def predictor_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Train predictor to match target; z is expected detached."""
        with torch.no_grad():
            t = self.target_embed(z)
        p = self.predict_embed(z)
        return F.mse_loss(p, t)


class IntrinsicValueHead(nn.Module):
    """Scalar value head for intrinsic returns"""
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.v = nn.Linear(feat_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.v(z).squeeze(-1)
