import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Dinov2ClassifierHead(nn.Module):
    def __init__(self, embed_dim, layers, checkpoint_path):
        super().__init__()
        self.layers = layers
        self.linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self.linear_head.load_state_dict(state_dict, strict=True)

    def forward(self, feature_list):
        # feature list: [[cls token, patch tokens], ..., [cls token, patch tokens]]
        x = feature_list
        if self.layers == 1:
            linear_input = torch.cat(
                [
                    x[0][0],  # cls token
                    x[0][1].mean(dim=1),  # patch tokens
                ],
                dim=1,
            )
        elif self.layers == 4:
            linear_input = torch.cat(
                [
                    x[0][0],  # cls token
                    x[1][0],  # cls token
                    x[2][0],  # cls token
                    x[3][0],  # cls token
                    x[3][1].mean(dim=1),  # patch tokens
                ],
                dim=1,
            )
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)

    def predict(self, feature_list, topk=1):
        logits = self.forward(feature_list)
        preds = F.softmax(logits, dim=1)
        values, indices = torch.topk(preds, topk, dim=1)
        return indices


class Dinov2SegmentationHead(nn.Module):
    """Batchnorm + Conv"""

    def __init__(
        self,
        in_channels,
        in_index,
        input_transform,
        channels,
        resize_factors=None,
        align_corners=False,
        num_classes=21,
        patch_size=16,
        dropout_ratio=0,
        checkpoint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.in_index = in_index
        self.input_transform = input_transform
        self.channels = channels
        self.resize_factors = resize_factors
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.align_corners = align_corners

        self.bn = nn.SyncBatchNorm(channels)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        if checkpoint is not None:
            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {
                k.replace("decode_head.", ""): v for k, v in state_dict.items()
            }
            self.load_state_dict(state_dict, strict=True)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.

            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (
                    len(self.resize_factors),
                    len(inputs),
                )
                inputs = [
                    resize(
                        input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area"
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            # actually no size is changed
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        assert inputs.shape[1] == self.channels, (
            f"Input channels {inputs.shape[1]} does not match expected channels {self.channels}"
        )
        return inputs

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv_seg(x)
        return x

    def predict(self, inputs, scale=1, size=None):
        seg_logits = self.forward(inputs)
        _, _, tok_h, tok_w = seg_logits.shape
        if scale != 1:
            seg_logits = resize(
                input=seg_logits,
                size=(tok_h * scale, tok_w * scale),
                mode="bilinear",
                align_corners=self.align_corners,
            )
        elif size is not None:
            assert isinstance(size, tuple)
            seg_logits = resize(
                input=seg_logits,
                size=size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return seg_logits

    def slide_predict(self, feature_list, current_size, slide_window, slide_stride, target_size=None):
        # feature_list: [[(B,C,H,W), ...], ..., [(B,C,H,W), ...]], 
        # which is N_Crop times N_Layer of features
        device = next(self.conv_seg.parameters()).device
        h_img, w_img = current_size
        h_stride, w_stride = slide_stride
        h_crop, w_crop = slide_window
        batch_size = feature_list[0][0].shape[0]
        num_classes = self.num_classes

        # Initialize predictions and counting matrix
        preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=device)
        count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=device)

        i = 0
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_seg_logit = self.predict(
                    feature_list[i], size=(h_crop, w_crop)
                )
                preds += F.pad(
                    crop_seg_logit, (x1, preds.shape[3] - x2, y1, preds.shape[2] - y2)
                )
                count_mat[:, :, y1:y2, x1:x2] += 1
                i += 1

        assert (count_mat == 0).sum() == 0, "Zero count in count matrix detected"
        preds = preds / count_mat
        if target_size is not None:
            preds = resize(
                input=preds,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return preds

