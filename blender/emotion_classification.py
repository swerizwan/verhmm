import torch
import torch.nn as nn

class EmotionClassificationModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2):
        """
        Initialize the emotion classification model.
        
        Args:
            input_size (int): Dimensionality of input features.
            output_size (int): Number of emotion classes for classification.
            hidden_size (int): Number of units in the hidden layers (default is 128).
            num_layers (int): Number of recurrent layers (default is 2).
        """
        super(EmotionClassificationModel, self).__init__()
        
        # Define classification model layers and architecture
        # LSTM is used to capture sequential information from input features.
        # The input size is the dimensionality of the input features.
        # The hidden size is the number of units in the hidden layers.
        # The number of layers specifies how many LSTM layers are stacked.
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to emotion probabilities.
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_features):
        """
        Forward pass for emotion classification.
        
        Args:
            input_features (torch.Tensor): Input features for classification.
            
        Returns:
            output_emotion_probabilities (torch.Tensor): Output emotion probabilities.
        """
        # Implement forward pass for classification
        # Pass input features through LSTM layers
        rnn_output, _ = self.rnn(input_features)
        
        # Get the output of the last time step
        last_output = rnn_output[:, -1, :]
        
        # Pass the last output through fully connected layer
        output_emotion_probabilities = self.fc(last_output)
        
        return output_emotion_probabilities
