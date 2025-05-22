import math
import torch
from torch import nn


class Sine(nn.Linear):
    """
        Sine activation function for linear layer
        Implementation based on https://github.com/vsitzmann/siren/blob/master/modules.py"""

    def __init__(self, in_features, out_features, bias=True, sine_fac=30.):
        super().__init__(in_features, out_features, bias)
        self.sine_fac = sine_fac

    def forward(self, x):
        return torch.sin(self.sine_fac * super().forward(x))


class Wire(nn.Module):
    """
        Wire activation function for linear layer
        Implementation based on "https://github.com/vishwa91/wire/blob/main/modules/wire.py"""
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0, trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        # Optionally: set requires_grad for trainable omega/scale if needed

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        return torch.cos(omega) * torch.exp(-(scale ** 2))


class SirenINR(nn.Module):
    """
    Implicit Neural Representation (INR) model using SIREN (Sinusoidal Representation Networks).
    This network uses sine activations and custom initialization as described in the SIREN paper.
    Implementation based on https://github.com/vsitzmann/siren/blob/master/modules.py,
    parts of the implementation are inspired by the INR Tutorial https://github.com/INR4MICCAI/INRTutorial

    Args:
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        hidden_size (int): Number of units in hidden layers.
        num_layers (int): Number of hidden layers.
        sine_fac (float): Frequency factor for sine activation.
        outermost_linear (bool): If True, last layer is linear, else uses sine.
    """
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=3,
                 sine_fac=30.0,
                 outermost_linear=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sine_fac = sine_fac
        self.outermost_linear = outermost_linear

        layers = []
        # First layer (special init)
        first = Sine(in_size, hidden_size, sine_fac=sine_fac)
        nn.init.uniform_(first.weight, -1 / in_size, 1 / in_size)
        layers.append(first)
        # Hidden layers
        for _ in range(num_layers):
            hidden = Sine(hidden_size, hidden_size, sine_fac=sine_fac)
            nn.init.uniform_(hidden.weight, -math.sqrt(6 / hidden_size) / sine_fac, math.sqrt(6 / hidden_size) / sine_fac)
            layers.append(hidden)
        # Output layer
        if outermost_linear:
            last = nn.Linear(hidden_size, out_size)
            nn.init.uniform_(last.weight, -math.sqrt(6 / hidden_size) / sine_fac, math.sqrt(6 / hidden_size) / sine_fac)
            layers.append(last)
        else:
            last = Sine(hidden_size, out_size, sine_fac=sine_fac)
            nn.init.uniform_(last.weight, -math.sqrt(6 / hidden_size) / sine_fac, math.sqrt(6 / hidden_size) / sine_fac)
            layers.append(last)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class WireINR(nn.Module):
    """
    Implicit Neural Representation (INR) model using Wire (Gabor) nonlinearity.
    This network uses the Wire (real Gabor) activation in all layers except the last, which is linear.
    Implementation based on https://github.com/vsitzmann/siren/blob/master/modules.py,
    parts of the implementation are inspired by the INR Tutorial https://github.com/INR4MICCAI/INRTutorial

    Args:
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        hidden_size (int): Number of units in hidden layers.
        num_layers (int): Number of hidden layers.
        wire_omega (float): Frequency factor for Gabor activation.
        wire_sigma (float): Scale factor for Gabor activation.
        outermost_linear (bool): If True, last layer is linear, else uses Wire.
    """
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=3,
                 wire_omega=10.0,
                 wire_sigma=10.0,
                 outermost_linear=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.wire_omega = wire_omega
        self.wire_sigma = wire_sigma
        self.outermost_linear = outermost_linear

        layers = []
        # First layer (special init)
        first = Wire(in_size, hidden_size, omega0=wire_omega, sigma0=wire_sigma, is_first=True)
        nn.init.uniform_(first.freqs.weight, -1 / in_size, 1 / in_size)
        nn.init.uniform_(first.scale.weight, -1 / in_size, 1 / in_size)
        layers.append(first)
        # Hidden layers
        for _ in range(num_layers):
            hidden = Wire(hidden_size, hidden_size, omega0=wire_omega, sigma0=wire_sigma)
            nn.init.uniform_(hidden.freqs.weight, -math.sqrt(6 / hidden_size) / wire_omega, math.sqrt(6 / hidden_size) / wire_omega)
            nn.init.uniform_(hidden.scale.weight, -math.sqrt(6 / hidden_size) / wire_omega, math.sqrt(6 / hidden_size) / wire_omega)
            layers.append(hidden)
        # Output layer
        if outermost_linear:
            last = nn.Linear(hidden_size, out_size)
            nn.init.uniform_(last.weight, -math.sqrt(6 / hidden_size) / wire_omega, math.sqrt(6 / hidden_size) / wire_omega)
            layers.append(last)
        else:
            last = Wire(hidden_size, out_size, omega0=wire_omega, sigma0=wire_sigma)
            nn.init.uniform_(last.freqs.weight, -math.sqrt(6 / hidden_size) / wire_omega, math.sqrt(6 / hidden_size) / wire_omega)
            nn.init.uniform_(last.scale.weight, -math.sqrt(6 / hidden_size) / wire_omega, math.sqrt(6 / hidden_size) / wire_omega)
            layers.append(last)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
