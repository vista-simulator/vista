from typing import Dict, List, Optional

import torch
from torch import nn

__all__ = ['Net']


class Net(nn.Module):
    def __init__(self, extractors: nn.ModuleDict, *,
                 lookaheads: Optional[List[int]] = [0]) -> None:
        super().__init__()
        self.extractors = extractors
        self.lookaheads = lookaheads

        feature_size = 0
        for extractor in self.extractors.values():
            feature_size += extractor.feature_size
        output_size = len(lookaheads)

        self.transform = nn.Sequential(
            nn.Linear(feature_size, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(100, output_size)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for name, extractor in self.extractors.items():
            feature = extractor(inputs[name])
            features.append(feature)
        outputs = torch.cat(features, dim=1)
        outputs = self.transform(outputs)
        return outputs
