import torch
import torch.nn as nn


class Dinov2ClassifierHead(nn.Module):
    def __init__(self, embed_dim, layers, checkpoint_path):
        super().__init__()
        self.layers = layers
        self.linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

        if checkpoint_path:
            print("Load head checkpoint from local path: ", checkpoint_path)
            state_dict = torch.load(checkpoint_path)
            self.linear_head.load_state_dict(state_dict, strict=True)

    def forward(self, feature_list):
        if self.layers == 1:
            x_norm = self.backbone.norm(feature_list[0])  # Perform norm in the head

            cls_token = x_norm[:, 0]
            patch_tokens = x_norm[:, self.backbone.num_register_tokens + 1 :]
            linear_input = torch.cat(
                [
                    cls_token,
                    patch_tokens.mean(dim=1),
                ],
                dim=1,
            )
        elif self.layers == 4:
            x = feature_list
            linear_input = torch.cat(
                [
                    x[0][1],
                    x[1][1],
                    x[2][1],
                    x[3][1],
                    x[3][0].mean(dim=1),
                ],
                dim=1,
            )
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)

    def predict(self, feature_list, topk=1):
        values, indices = torch.topk(self.forward(feature_list), topk, dim=1)
        return indices
