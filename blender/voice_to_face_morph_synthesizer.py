import torch
import torch.nn as nn

class VoiceToFaceMorphSynthesizer(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, hidden_size=256):
        """
        Initialize the voice to face morph synthesizer.
        
        Args:
            input_size (int): Dimensionality of input features.
            output_size (int): Dimensionality of output morph coefficients.
            num_layers (int): Number of transformer layers (default is 2).
            hidden_size (int): Number of units in the hidden layers (default is 256).
        """
        super(VoiceToFaceMorphSynthesizer, self).__init__()
        
        # Define transformer layers and architecture
        # Transformer layers are used to capture dependencies between input features.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dim_feedforward=hidden_size),
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size)
        ])
        # Transformer Encoder processes the input sequence using the specified number of layers.
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)
        
        # Fully connected layer to map transformer output to morph coefficients.
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_features):
        """
        Forward pass through the voice to face morph synthesizer.
        
        Args:
            input_features (torch.Tensor): Input features for the synthesizer.
            
        Returns:
            output_morph_coefficients (torch.Tensor): Output morph coefficients.
        """
        # Forward pass through the transformer encoder
        transformer_output = self.transformer_encoder(input_features)
        
        # Map the transformer output to morph coefficients using the fully connected layer.
        output_morph_coefficients = self.fc(transformer_output)
        
        return output_morph_coefficients
