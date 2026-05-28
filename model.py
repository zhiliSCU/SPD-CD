import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 1. PRMC: Physics-Driven Radiometric Mapping Compensation
# ================================================================

class PRMC(nn.Module):
    """
    Physics-Driven Radiometric Mapping Compensation (PRMC)

    Designed to:
        - Model terrain-induced radiometric distortion
        - Separate high-frequency curvature components
        - Apply curvature sparsity regularization
        - Restore cross-temporal radiometric consistency
    """

    def __init__(self, channels: int):
        super().__init__()

        # Physics-inspired curvature operator
        self.curvature_operator = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        self._initialize_operator()

        # Learnable sparsity threshold
        self.tau = nn.Parameter(torch.tensor(0.1))

        # Radiometric mapping network
        self.mapping = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1)
        )

    def _initialize_operator(self):
        """
        Initialize curvature operator.

        The specific physical kernel design
        is abstracted in this conceptual release.
        """
        with torch.no_grad():
            self.curvature_operator.weight.zero_()

        self.curvature_operator.weight.requires_grad = False

    def forward(self, x):

        curvature = self.curvature_operator(x)

        tau = torch.clamp(self.tau, min=1e-6)

        sparse_curvature = torch.sign(curvature) * F.relu(
            torch.abs(curvature) - tau
        )

        compensated = self.mapping(
            x - curvature + sparse_curvature
        )

        curvature_loss = torch.mean(torch.abs(sparse_curvature))

        return compensated, curvature_loss


# ================================================================
# 2. CRIFA: Cross-Resolution Implicit Feature Alignment
# ================================================================

class CRIFA(nn.Module):
    """
    Cross-Resolution Implicit Feature Alignment (CRIFA)

    Instead of explicit geometric registration,
    multi-scale local correlation matrices are
    constructed in feature space to achieve
    cross-resolution alignment.
    """

    def __init__(self):
        super().__init__()

    def compute_correlation(self, query, key):

        scale = query.shape[-1] ** 0.5

        correlation = torch.bmm(
            query / scale,
            key.transpose(-2, -1)
        )

        return torch.softmax(correlation, dim=-1)

    def forward(self, feat_low_res, feat_high_res):
        """
        Conceptual forward.

        Multi-scale partition strategy and
        feature reconstruction steps are omitted.
        """

        B, C, H, W = feat_low_res.shape

        query = feat_low_res.flatten(2).transpose(1, 2)
        key = feat_high_res.flatten(2).transpose(1, 2)

        correlation_matrix = self.compute_correlation(query, key)

        return correlation_matrix


# ================================================================
# 3. Differential Change Decoder
# ================================================================

class DifferentialDecoder(nn.Module):
    """
    Differential change decoder.

    Generates pixel-level change maps
    from aligned feature representations.
    """

    def __init__(self, channels=8, num_classes=2):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_classes, 1)
        )

    def forward(self, x):
        return self.decoder(x)


# ================================================================
# 4. SPD-CD Framework
# ================================================================

class SPDCD(nn.Module):
    """
    Single-Stage Physics-Driven Cross-Resolution
    Change Detection Framework (SPD-CD)

    Resolves the registration–detection conflict
    via physics-driven radiometric modeling
    and implicit feature alignment.
    """

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.prmc = PRMC(channels=8)
        self.crifa = CRIFA()
        self.decoder = DifferentialDecoder(channels=8)

    def forward(self, img_pre, img_post):

        # Feature extraction
        feat_pre = self.backbone(img_pre)
        feat_post = self.backbone(img_post)

        # Physics-driven radiometric compensation
        compensated_feat, curvature_loss = self.prmc(feat_pre)

        # Implicit cross-resolution alignment
        correlation = self.crifa(
            compensated_feat,
            feat_post
        )

        # Differential representation (abstracted)
        diff_feature = torch.abs(compensated_feat - feat_post)

        # Change map prediction
        prediction = self.decoder(diff_feature)

        return {
            "prediction": prediction,
            "curvature_regularization": curvature_loss,
            "correlation_matrix": correlation
        }