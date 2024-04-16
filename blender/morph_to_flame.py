import torch

def morph_to_flame(morph_coefficients, flame_templates):
    """
    Transform morph coefficients to FLAME parameters.
    
    Args:
        morph_coefficients (torch.Tensor): Morph coefficients.
        flame_templates (torch.Tensor): FLAME head templates.
        
    Returns:
        flame_parameters (torch.Tensor): FLAME parameters.
    """
    # Ensure input tensors are of the correct shape
    assert morph_coefficients.shape[1] == flame_templates.shape[0], "Dimensions of input tensors are not compatible."
    
    # Apply linear weighting to morph coefficients
    # Each row of flame_templates represents a FLAME head template
    # Each column of morph_coefficients represents a coefficient for blending the templates
    # The dot product of morph_coefficients and flame_templates performs the linear weighting
    flame_parameters = torch.matmul(morph_coefficients, flame_templates)
    
    return flame_parameters
