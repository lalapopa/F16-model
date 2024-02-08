import numpy as np
import matplotlib.pyplot as plt
import random


class ReferenceSignal:
    """Docstring for ReferenceSignal."""

    def __init__(self, t0, dt, tn, determenistic=False):
        self.t0 = t0
        self.dt = dt
        self.tn = tn
        self.t = np.arange(self.t0, self.tn, self.dt)
        self.theta_ref = np.ones(self.t.shape)
        self.determenistic = determenistic
        self.get_reference_signal()

    def get_reference_signal(self):
        if self.determenistic:
            A_theta = 10 * np.pi / 180  # [rad]
        else:
            A_theta = np.radians(random.choice([20, 10, -10, -20]))

        self.theta_ref = A_theta * self.cosstep(-1, 2) * 2 - A_theta
        self.theta_ref -= A_theta * self.cosstep(self.tn * 0.25, 1)
        self.theta_ref -= A_theta * self.cosstep(self.tn * 0.50, 1)
        self.theta_ref += A_theta * self.cosstep(self.tn * 0.75, 1)
        self.theta_ref = self.theta_ref[
            ::-1
        ]  # Flip ref signal for zero in first 2 second to prevent tilt in begining

    def cosstep(self, start_time, w):
        """
        Smooth cosine step function starting at t0 with width w
        """

        t0_idx = np.abs(self.t - start_time).argmin()
        t1 = start_time + w
        t1_idx = np.abs(self.t - t1).argmin()

        a = -(np.cos(1 / w * np.pi * (self.t - start_time)) - 1) / 2
        if start_time >= 0:
            a[:t0_idx] = 0.0
        a[t1_idx:] = 1.0
        return a


if __name__ == "__main__":
    ref_signal = ReferenceSignal(0, 0.01, 10)
    plt.plot(ref_signal.t, ref_signal.theta_ref)
    plt.show()
