import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    """
    Co-Attention module for multimodal fusion of text, audio, and video features.
    """
    def __init__(self, text_dim, audio_dim, video_dim, hidden_dim):
        super(CoAttention, self).__init__()

        # Projection layers for each modality
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.video_projection = nn.Linear(video_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

        # Self-attention layers for each modality
        self.self_attention = nn.Linear(hidden_dim, 1)

        # Cross-modal attention layers
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, text_features, audio_features, video_features):
        # Project features
        text_proj = self.relu(self.layer_norm(self.text_projection(text_features)))
        audio_proj = self.relu(self.layer_norm(self.audio_projection(audio_features)))
        video_proj = self.relu(self.layer_norm(self.video_projection(video_features)))

        # Compute self-attention scores
        text_attn = torch.softmax(self.self_attention(text_proj), dim=1)
        audio_attn = torch.softmax(self.self_attention(audio_proj), dim=1)
        video_attn = torch.softmax(self.self_attention(video_proj), dim=1)

        # Apply attention weights
        text_proj = text_proj * text_attn
        audio_proj = audio_proj * audio_attn
        video_proj = video_proj * video_attn

        # Cross-modal attention
        text_audio_attn, _ = self.cross_attention(text_proj, audio_proj, audio_proj)
        audio_video_attn, _ = self.cross_attention(audio_proj, video_proj, video_proj)
        video_text_attn, _ = self.cross_attention(video_proj, text_proj, text_proj)

        # Concatenate attended features
        combined = torch.cat([text_audio_attn, audio_video_attn, video_text_attn], dim=-1)

        # Final fusion
        fused_output = self.fusion(combined)

        return fused_output


class ContextualFusionCoAttention(nn.Module):
    """
    Contextual Fusion Model using Co-Attention and Transformer Encoder.
    """
    def __init__(self, text_dim, audio_dim, video_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(ContextualFusionCoAttention, self).__init__()

        # Two-layer Co-Attention mechanism
        self.co_attention_1 = CoAttention(text_dim, audio_dim, video_dim, hidden_dim)
        self.co_attention_2 = CoAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim)

        # Residual Gating Mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification layer
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 5)  # 5 sentiment classes: Negative, Neutral, Positive, Strong Negative, Strong Positive
        )

    def forward(self, text_features, audio_features, video_features):
        # Apply first Co-Attention layer
        fused_output = self.co_attention_1(text_features, audio_features, video_features)

        # Apply second Co-Attention layer
        fused_output = self.co_attention_2(fused_output, fused_output, fused_output)

        # Apply residual gating
        gated_output = self.gate(fused_output) * fused_output

        # Transformer Encoder expects (batch, seq_len, hidden_dim) -> Add sequence dimension
        gated_output = gated_output.unsqueeze(1)

        # Apply Transformer Encoder
        encoded_output = self.transformer_encoder(gated_output)

        # Extract last timestep output for classification
        final_output = self.fc_final(encoded_output[:, -1, :])

        return final_output
