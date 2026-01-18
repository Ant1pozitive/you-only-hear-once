import torch
import torch.nn as nn

class AudioMemoryBank(nn.Module):
    """Adapted from memory-is-all-you-need for audio continual learning."""
    def __init__(self, num_slots: int, slot_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.01)
        self.synthesizer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(slot_dim, nhead=4, dim_feedforward=slot_dim*2),
            num_layers=2
        )

    def read(self, query: torch.Tensor) -> torch.Tensor:
        # Dot-product attention for retrieval
        sim = torch.einsum("btd,bsd->bts", query, self.slots)  # [B, T, Slots]
        weights = nn.functional.softmax(sim, dim=-1)
        return torch.einsum("bts,bsd->btd", weights, self.slots)

    def write(self, key: torch.Tensor, value: torch.Tensor):
        # Simple Hebbian update (simplified)
        update = torch.einsum("btd,btd->bd", key, value)
        self.slots.data += 0.01 * update.mean(0, keepdim=True)  # Global update

    def synthesize(self):
        # Dreaming: self-attention on slots
        self.slots.data = self.synthesizer(self.slots)
