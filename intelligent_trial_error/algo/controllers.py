
"""
Main hexapod open-loop controller definition.

References:
- https://github.com/resibots/pycontrollers

"""


import numpy as np
import math


class OpenLoopController:
    """
    Implement an open-loop controller based on periodic signals
    Please see the supplementary information of Cully et al., Nature, 2015
    """

    def __init__(self, array_dim=100):
        self.array_dim = array_dim
        self.trajs = np.zeros(1)
        # Set random seed
        np.random.seed(100)

    # def step(self, simu):
    #     assert(self.trajs.shape[0] != 1)
    #     k = int(math.floor(simu.t * self.array_dim)) % self.array_dim
    #     return self.trajs[:, k]

    def get_action(self, state):
        """Provide an appropriate action based on the current timestep"""
        timestep = state[0]
        assert(self.trajs.shape[0] != 1)
        k = int(math.floor(timestep * self.array_dim)) % self.array_dim
        return self.trajs[:, k]

    def _control_signal(self, amplitude, phase, duty_cycle, array_dim=100):
        """
        Create a smooth periodic function with amplitude, phase, and duty cycle,
        amplitude, phase and duty cycle are in [0, 1].
        These are based on the parameter vector provided to the controller.
        """
        assert(amplitude >= 0 and amplitude <= 1)
        assert(phase >= 0 and phase <= 1)
        assert(duty_cycle >= 0 and duty_cycle <= 1)
        command = np.zeros(array_dim)

        # create a 'top-hat function'
        up_time = array_dim * duty_cycle
        temp = [
            amplitude if i < up_time else -amplitude
            for i in range(0, array_dim)]

        # smoothing kernel
        kernel_size = int(array_dim / 10)
        kernel = np.zeros(int(2 * kernel_size + 1))
        sigma = kernel_size / 3
        for i in range(0, len(kernel)):
            kernel[i] = math.exp(
                - (i - kernel_size) * (i - kernel_size) / (2 * sigma**2)) \
                / (sigma * math.sqrt(math.pi))
        sum = np.sum(kernel)

        # smooth the function
        for i in range(0, array_dim):
            command[i] = 0
            for d in range(1, kernel_size + 1):
                if i - d < 0:
                    command[i] += temp[array_dim + i - d] \
                        * kernel[kernel_size - d]
                else:
                    command[i] += temp[i - d] * kernel[kernel_size - d]
            command[i] += temp[i] * kernel[kernel_size]

            for d in range(1, kernel_size + 1):
                if i + d >= array_dim:
                    command[i] += temp[i + d - array_dim] \
                        * kernel[kernel_size + d]
                else:
                    command[i] += temp[i + d] * kernel[kernel_size + d]
            command[i] /= sum

        # shift according to the phase
        final_command = np.zeros(array_dim)
        start = int(math.floor(array_dim * phase))
        current = 0
        for i in range(start, array_dim):
            final_command[current] = command[i]
            current += 1
        for i in range(0, start):
            final_command[current] = command[i]
            current += 1

        assert(len(final_command) == array_dim)

        return final_command


class HexapodController(OpenLoopController):
    """
    This should be the same controller as Cully et al., Nature, 2015

    Create a control signal for each of the leg joints (rows) for all the
    timesteps (columns).
    """

    def __init__(self, parameter_size=36, parameter_range=(0,1), array_dim=100):
        super(HexapodController, self).__init__(array_dim)
        self.parameter_size = parameter_size
        self.parameter_range = parameter_range

    def _compute_trajs(self, params, array_dim):
        trajs = np.zeros((4 * 3, array_dim))
        k = 0
        for i in range(0, 36, 9):
            trajs[k, :] = ((math.pi * self._control_signal(
                params[i], params[i + 1], params[i + 2], array_dim)) - math.pi * 0.5) * 0.125
            trajs[k + 1, :] = ((math.pi * self._control_signal(
                params[i], params[i + 1], params[i + 2], array_dim)) - math.pi * 0.5) * 0.0625 + 0.75
            trajs[k + 2, :] = ((math.pi * self._control_signal(
                params[i], params[i + 1], params[i + 2], array_dim)) - math.pi * 0.5) * 0.0625 + 1.45
            k += 3
        return trajs

    def set_parameters(self, ctrl_parameters):
        """Update the controlloer parameters and recompute the trajectory."""
        self.params = ctrl_parameters
        self.trajs = self._compute_trajs(ctrl_parameters, self.array_dim)

    def generate_parameters(self, n_samples):
        """Generate parameters randomly."""
        return np.random.uniform(
            *self.parameter_range, size=(n_samples, self.parameter_size))
