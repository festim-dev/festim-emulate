import numpy as np

from autoemulate.simulations.base import Simulator
import torch


class SpectrumSimple(Simulator):
    """
    Simulator of a spectrum made of 2 gaussian peaks.
    """

    def __init__(self, param_ranges, output_names, times=None):
        super().__init__(param_ranges, output_names, log_level="error")
        if times is None:
            times = torch.linspace(450, 500, 20)
        self.times = times

    def _forward(self, x):
        """
        Calculate the horizontal distance a projectile travels using PyTorch.

        Parameters:
        ----------
        x0 : torch.Tensor
            Position of first peak in meters.
        height0 : torch.Tensor
            Height of first peak.
        x1 : torch.Tensor
            Position of second peak in meters.
        height1 : torch.Tensor
            Height of second peak.

        Returns:
        -------
        torch.Tensor
            Spectrum intensity values.
        """
        mean0 = x[:, 0]
        height0 = x[:, 1]
        mean1 = x[:, 2]
        height1 = x[:, 3]
        stddev = 10

        # Reshape times to be a column vector for proper broadcasting
        times_col = self.times.unsqueeze(1)  # Shape: (500, 1)

        # first gaussian peak
        peak1 = height0 * torch.exp(-0.5 * ((times_col - mean0) / stddev) ** 2)
        # second gaussian peak
        peak2 = height1 * torch.exp(-0.5 * ((times_col - mean1) / stddev) ** 2)

        # sum of both peaks
        desorption = peak1 + peak2  # Shape: (500, batch_size)

        # Transpose to get (batch_size, 500) - rows are samples, columns are outputs
        desorption = desorption.T

        return desorption


def simulate(x):
    time = np.linspace(450, 500, 500)

    x0, h1, x1, h2 = x
    stddev = 10
    # first gaussian peak
    peak1 = h1 * np.exp(-0.5 * ((time - x0) / stddev) ** 2)
    # second gaussian peak
    peak2 = h2 * np.exp(-0.5 * ((time - x1) / stddev) ** 2)

    desorption = peak1 + peak2

    return desorption, time
