import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import copy
    
# class Extractor(nn.Module):
#     def __init__(self, pretrained_model, ):
#         super(Extractor, self).__init__()
#         self.extractor = pretrained_model

#     def forward(self, x):
#         x = self.extractor(x)
#         return x
    
# class Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(Classifier, self).__init__()
#         self.logits = nn.Sequential(
#             nn.Linear(512, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.8),
#             nn.Linear(4096, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.logits(x)
#         return x

class AugmentationLayer(nn.Module):
    def __init__(self, p=0.3, noise_std=0.05):
        """
        Args:
        p: Percentage of channels to be augmented (0 <= p <= 1)
        """
        super(AugmentationLayer, self).__init__()
        self.p = p
        self.noise_std = noise_std
        
        # Các phép augment dành cho feature maps
        self.augment_transforms = nn.ModuleList([
            transforms.RandomResizedCrop((3, 3)),    
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomRotation(180),           
            transforms.GaussianBlur(kernel_size=(3, 3))  
        ])

    def forward(self, x):
        """
        x: Input feature map of shape (b, c, h, w)
        Returns:
        x': Augmented feature map
        """
        batch_size, num_channels, h, w = x.size()

        num_aug_channels = int(self.p * num_channels)
        aug_indices = torch.randperm(num_channels)[:num_aug_channels]

        for idx in aug_indices:
            channel = x[:, idx]

            for aug_transform in self.augment_transforms:
                channel = aug_transform(channel)

            x[:, idx] = channel 

        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                # The backbone will throw an Exception once it reached the last
                # layer to compute features from. Computation will stop there.
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

class Preprocessing(torch.nn.Module):
    def __init__(self, output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        module = MeanMapper(output_dim)
        self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
    
class FaceNetClassifierWithAugFMap(nn.Module):
    def __init__(self, num_classes, feature_agg, aug_layer = None):
        super(FaceNetClassifierWithAugFMap, self).__init__()
        self.extractor = feature_agg
        self.aug_layer = None if aug_layer is None else aug_layer
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1792, num_classes)
        )
    
    def forward(self, x):
        features = self.extractor(x)['block8']
        if self.aug_layer is not None:
            features = self.aug_layer(features)
        # features = features.view(features.size(0), -1)
        x = self.fc(features)
        return x
    

if __name__ == "__main__":
    # Sử dụng Augmentation Layer trong quá trình training
    augment_layer = AugmentationLayer(p=0.3, noise_std=0.05)

    # Giả sử feature map có kích thước (batch_size, num_channels, height, width)
    feature_map = torch.randn(64, 1792, 3, 3)

    # Augment feature map
    augmented_feature_map = augment_layer(feature_map)
    print(augmented_feature_map.shape)  # torch.Size([64, 1792, 3, 3])