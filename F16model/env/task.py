import numpy as np
import matplotlib.pyplot as plt
import random


class ReferenceSignal:
    """Class where reference signal is build"""

    def __init__(self, dt, tn, determenistic=False, scenario=None):
        self.dt = dt
        self.tn = tn
        self.t = np.arange(0, self.tn, self.dt)
        self.determenistic = determenistic
        if scenario == None:
            scenario = random.choice(["step", "cos", "pure_step"])

        if scenario == "step":
            reference_signal = self.get_step_reference_signal()
        elif scenario == "cos":
            reference_signal = self.get_cos_reference_signal()
        elif scenario == "pure_step":
            reference_signal = self.get_pure_step_reference_signal()
        elif scenario == "combo":
            reference_signal = self.get_combo_reference_signal()
        self.theta_ref = reference_signal
        self.wz_ref = reference_signal / 2

    def get_step_reference_signal(self):
        if self.determenistic:
            A_theta = 10 * np.pi / 180  # [rad]
        else:
            A_theta = np.radians(random.choice([20, 10, 5, -5, -10, -20]))

        reference_signal = A_theta * self.cosstep(-1, 2) * 2 - A_theta
        reference_signal -= A_theta * self.cosstep(self.tn * 0.25, 1)
        reference_signal -= A_theta * self.cosstep(self.tn * 0.50, 1)
        reference_signal += A_theta * self.cosstep(self.tn * 0.75, 1)
        reference_signal = reference_signal[
            ::-1
        ]  # Flip ref signal for zero in first 2 second to prevent Agent tilt in begining
        return reference_signal

    def get_cos_reference_signal(self):
        if self.determenistic:
            A_theta = 20 * np.pi / 180  # [rad]
            freq = 0.25
        else:
            A_theta = np.radians(random.choice([10, 5, -5, -10]))
            freq = random.choice([0.125, 0.25, 0.45, 0.5])
        reference_signal = np.sin(2 * np.pi * freq * self.t) * A_theta
        return reference_signal

    def get_pure_step_reference_signal(self):
        reference_signal = np.zeros(self.t.shape)
        if self.determenistic:
            amp = np.radians(10)
            t_step = 1
        else:
            amp = np.radians(random.choice([20, 10, 5, -5, -10, -20]))
            t_step = random.choice([1, 2, 3, 4])
        t_step_idx = np.abs(self.t - t_step).argmin()
        reference_signal[t_step_idx:] = amp
        return reference_signal

    def get_combo_reference_signal(self):
        reference_signal = np.zeros(self.t.shape)
        if self.determenistic:
            step_amp = np.radians(10)
            sin_amp = np.radians(10)
            freq = 0.25
            cos_step_amp = np.radians(20)
        else:
            step_amp = np.radians(random.choice([20, 10, 5, -5, -10, -20]))
            sin_amp = np.radians(random.choice([20, 10, 5, -5, -10, -20]))
            freq = random.choice([0.125, 0.25, 0.45, 0.5, 0.75])
            cos_step_amp = np.radians(random.choice([20, 10, 5, -5, -10, -20]))

        scenario = []
        for _ in range(0, int(self.tn / 10)):
            signal_len = int(1 / self.dt)
            step_signal = np.concatenate(
                (
                    [0] * int(signal_len / 2),
                    [step_amp] * int(signal_len * 2),
                    [0] * int(signal_len / 2),
                )
            )
            sin_signal = np.sin(2 * np.pi * freq * np.arange(0, 3, self.dt)) * sin_amp
            cos_step_signal = np.concatenate(
                (
                    self.cosstep(0, 1)[:signal_len] * cos_step_amp,
                    [cos_step_amp] * signal_len,
                    self.cosstep(0, 1)[:signal_len][::-1] * cos_step_amp,
                )
            )
            scenario.append(step_signal)
            scenario.append(sin_signal)
            scenario.append(cos_step_signal)
        if not self.determenistic:
            random.shuffle(scenario)
        scenario = np.concatenate(scenario)
        reference_signal[: len(scenario)] = scenario
        return reference_signal

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
    ref_signal = ReferenceSignal(0.01, 10, determenistic=False, scenario="step")
    plt.plot(ref_signal.t, ref_signal.theta_ref)
    plt.show()
