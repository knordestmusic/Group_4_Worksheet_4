import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd


class waveGuide:

    def __init__(self, L, pickup_position, loss_factor=0.995, filter_size=6):
        self.fs = 48000
        self.L = L  # length of the buffer
        self.pickup_position = max(
            1, min(pickup_position, self.L - 2)
        )  # clamp pickup position to between 1 and L-2

        self.g = loss_factor
        self.filter_size = filter_size

        # initializing the buffers
        self.x_L = np.zeros(self.L)
        self.x_R = np.zeros(self.L)

        self.filter = np.zeros(self.filter_size)

    def pluck(self, position):

        # clamp pluck position
        position = max(1, min(position, self.L - 2))

        # reset delay lines and filters
        self.filter = np.zeros(self.filter_size)
        self.x_L = np.zeros(self.L)
        self.x_R = np.zeros(self.L)

        # rising of the envelope
        self.x_L[1 : position + 1] = np.linspace(0, 1, position)
        self.x_R[1 : position + 1] = np.linspace(0, 1, position)

        # falling of the envelope

        self.x_L[position : self.L - 1] = np.linspace(1, 0, self.L - position - 1)
        self.x_R[position : self.L - 1] = np.linspace(1, 0, self.L - position - 1)

    def wavePropagation(self):

        # Get samples at boundaries
        l_out = self.x_L[0]  # Sample at the bridge (left end)
        r_out = self.x_R[self.L - 1]  # Sample at the nut (right end)

        # Calculate filter output (moving average)
        f_out = np.mean(self.filter)

        # Shift the values in the delay line to simulate the wave propogation
        self.x_L = np.roll(self.x_L, -1)
        self.x_R = np.roll(self.x_R, 1)

        # The values in the filter are shifted too
        self.filter = np.roll(self.filter, 1)

        # Reflection at the bridge
        self.x_L[self.L - 1] = -self.g * r_out

        # Reflection at the nut
        self.x_R[0] = -self.g * l_out

        # Fill the filter
        self.filter[0] = r_out

        # Read the output of the bi-directional buffer at the pickup_position

        out = self.x_L[self.pickup_position] + self.x_R[self.pickup_position]

        return out
