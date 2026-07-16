from .load import Database
from .sort import Recording
import os
from scipy.io.wavfile import WavFileWarning
import warnings

warnings.filterwarnings("ignore", category=WavFileWarning)


class Neuron:
    def __init__(self, session_id, unit_id, db: Database, rec: Recording):
        self.session_id = session_id
        self.unit_id = unit_id
        self.db = db
        self.rec = rec

        self.spiketimes = None
        self.stimuli = None
        self.padding = None
        self.neuron_data = None

        self.stimulus_data = None
        self.trial_psth = None
        self.baseline_psth = None

        self.figure_size = (8, 2)
        self.figures = {}

        self.padding = 0.5

        self.manual_area_curation_path = os.path.join(self.db.db_path, '..', 'neuron_locations.json')

    @property
    def load(self):
        import sqlite3, os
        import pandas as pd

        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        command = """
                  SELECT *
                  FROM neurons
                  WHERE session_id = ? \
                    AND unit_id = ? \
                  """
        cursor.execute(command, (self.session_id, self.unit_id))
        row = cursor.fetchone()

        # convert to dict
        self.neuron_data = dict(zip([col[0] for col in cursor.description], row))
        conn.close()

        spikes_path = self.neuron_data['spike_file']
        stim_path = self.neuron_data['stimulus_file']
        stim_path = os.path.join(stim_path, '..', 'stimuli.json')

        # load spikes & stims
        self.spiketimes = self.load_spiketimes(spikes_path, self.unit_id) / self.rec.samplerate
        self.stimuli = self.load_stimuli(stim_path)

        return pd.unique(self.stimuli['Stimuli Type'])

    @staticmethod
    def load_waveforms(waveforms_path, unit_id):
        import h5py

        data_dict = {}
        with h5py.File(waveforms_path, 'r') as f:
            mean = f['units'][f'unit_{unit_id}']['mean'][:]
            sd = f['units'][f'unit_{unit_id}']['sd'][:]
            width_pp = f['units'][f'unit_{unit_id}']['spike_width_pp'][()]
            width_hw = f['units'][f'unit_{unit_id}']['spike_width_hw'][()]
            return mean, sd, width_pp, width_hw

    @staticmethod
    def load_spiketimes(spikes_path, unit_id):
        import h5py

        data_dict = {}
        with h5py.File(spikes_path, "r") as f:
            return f[f'unit_{unit_id}'][:]

    @staticmethod
    def load_stimuli(stim_path):
        import pandas as pd

        return pd.read_json(stim_path)

    def plot(self, raster=True, psth=True, baseline=False, stimuli_to_raster=None, padding=0.5, normalize_psth=True,
             psth_sigma=0.015, psth_dt=0.001, raw_data=False, show=True, temporal_pcc=False):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self.padding = padding

        if stimuli_to_raster is None:
            stimuli_to_raster = pd.unique(self.stimuli['Stimuli Type'])

        num_rows = 1  # waveform always
        if raster:
            num_rows += 1
        if psth:
            num_rows += 1
        if baseline and raster:
            num_rows += 1
        if raw_data:
            num_rows += 1
        if temporal_pcc:
            num_rows += 1

        for s, stimulus in enumerate(stimuli_to_raster):
            fig = plt.figure(figsize=(8, 2 * num_rows))
            gs = gridspec.GridSpec(num_rows, 1, figure=fig, hspace=0.5, wspace=0.3)

            axes = []
            row_idx = 1
            ax_wave = fig.add_subplot(gs[0, 0])
            ax_wave.set_ylabel("Stimulus Amplitude (a.u.)")
            axes.append(ax_wave)
            _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, ax=ax_wave)
            if raw_data:
                ax_raw = fig.add_subplot(gs[row_idx, 0], sharex=ax_wave)
                axes.append(ax_raw)
                row_idx += 1
            if raster:
                ax_raster = fig.add_subplot(gs[row_idx, 0], sharex=ax_wave)
                axes.append(ax_raster)
                row_idx += 1
            if baseline and raster:
                ax_baseline = fig.add_subplot(gs[row_idx, 0], sharex=ax_wave)
                axes.append(ax_baseline)
                row_idx += 1
            if psth:
                ax_psth = fig.add_subplot(gs[row_idx, 0], sharex=ax_wave)
                axes.append(ax_psth)
                row_idx += 1
            if temporal_pcc:
                ax_pcc = fig.add_subplot(gs[row_idx, 0], sharex=ax_wave)
                row_idx += 1

            trial_spikes = None
            baseline_spikes = None

            if raw_data:
                _, _ = self.plot_raw_data(stimulus, ax=ax_raw)

            if raster or psth:
                _, trial_spikes, trial_duration = self.raster(
                    stimulus,
                    ax=ax_raster,
                    baseline=False,
                    plot=raster,
                    color='blue'
                )

            if baseline:
                _, baseline_spikes, trial_duration = self.raster(
                    stimulus,
                    ax=ax_baseline,
                    baseline=True,
                    plot=raster,
                    color='red',
                    padding=padding
                )

            if psth and trial_spikes is not None:
                _, self.trial_psth = self.psth(
                    stimulus,
                    trial_spikes,
                    ax=ax_psth,
                    normalize=normalize_psth,
                    dt=psth_dt,
                    sigma=psth_sigma,
                    color='blue'
                )

            if psth and baseline and baseline_spikes is not None:
                _, self.baseline_psth = self.psth(
                    stimulus,
                    baseline_spikes,
                    ax=ax_psth,
                    normalize=normalize_psth,
                    dt=psth_dt,
                    sigma=psth_sigma,
                    color='red'
                )

            if temporal_pcc:
                self.plot_temporal_pcc(stimulus, ax=ax_pcc)

            # axes = self.format_axes(axes, stimulus, baseline=(baseline and raster))
            fig.suptitle(f"Response of s.u. {self.unit_id} to {stimulus.strip(".wav")}")
            self.figures[stimulus] = fig

            if show:
                plt.show()

    def plot_response_strength(self, stimulus, color='blue', padding=0.18 * 2, save_path=None, name='default'):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, figure=fig)

        axes = []
        ax_raster = fig.add_subplot(gs[0, 0])
        axes.append(ax_raster)

        ax_rs = fig.add_subplot(gs[1, 0], sharex=ax_raster)
        axes.append(ax_rs)

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            ax=ax_raster,
            baseline=False,
            plot=True,
            color=color
        )
        _, baseline_spikes, baseline_duration = self.raster(
            stimulus,
            ax=None,
            baseline=True,
            plot=False
        )
        _, [trial_time, trial_psth] = self.psth(
            stimulus,
            trial_spikes,
            ax=None,
            plot=False,
            dt=0.001,
            sigma=0.015
        )
        _, [bl_time, bl_psth] = self.psth(
            stimulus,
            baseline_spikes,
            ax=None,
            plot=False,
            dt=0.001,
            sigma=0.015
        )
        rs = trial_psth - bl_psth

        ax_rs.plot(trial_time, rs, color=color)

        if save_path is not None:
            p = os.path.join(save_path, f'{name}.svg')
            plt.savefig(p, format='svg')
        plt.show()

    def plot_stimulus(self, stimulus, ax=None, plot=True, padding=None):
        import matplotlib.pyplot as plt
        import scipy.io.wavfile as wav
        import os
        import numpy as np

        if padding is None:
            padding = self.padding

        if plot and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        stim_path = os.path.join(self.db.stim_library, f'{stimulus}.wav')

        sampling_rate, data = wav.read(stim_path)
        data = data.astype(float)
        data /= max(np.abs(data))

        pad_samples = int(padding * sampling_rate)
        padded = np.concatenate([
            np.zeros(pad_samples),
            data,
            np.zeros(pad_samples)
        ])

        t = np.arange(len(padded)) / sampling_rate

        if plot:
            ax.plot(t - padding, padded, color='black', alpha=0.5)

        return ax, t, padded

    def raster(self, stimulus, baseline=False, plot=True, ax=None, color='black', padding=None, separate_trials=False):
        if padding == None:
            padding = self.padding

        if plot and ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        stim_times = self.stimuli[self.stimuli['Stimuli Type'] == stimulus].reset_index(drop=True)
        avg_duration = stim_times['Duration'].mean()
        all_trial_spikes, separate_trial_spikes = [], []
        for i, iteration in stim_times.iterrows():
            stim_start_time = iteration['Start Time']
            stim_end_time = iteration['End Time']

            if baseline:
                baseline_start = stim_start_time - avg_duration - 1

                mask = (
                        (self.spiketimes > baseline_start - padding) &
                        (self.spiketimes <= baseline_start + avg_duration + padding)
                )

                trial_spikes = self.spiketimes[mask] - baseline_start

            else:
                mask = (
                        (self.spiketimes > stim_start_time - padding) &
                        (self.spiketimes <= stim_end_time + padding)
                )

                trial_spikes = self.spiketimes[mask] - stim_start_time
            all_trial_spikes.extend(trial_spikes)
            separate_trial_spikes.append(trial_spikes)

            if plot:
                ax.vlines(trial_spikes, ymin=i, ymax=i + 0.5, color=color)
                ax.set_ylabel("Trial Number")
                ax.set_ylim(0, len(stim_times))

        if separate_trials:
            return ax, separate_trial_spikes, avg_duration

        return ax, all_trial_spikes, avg_duration

    def psth(self, stimulus, spikes, ax=None, sigma=0.015, dt=0.001, normalize=True, color='black', plot=True,
             padding=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        if padding is None:
            padding = self.padding

        if plot and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        stim_times = self.stimuli[self.stimuli['Stimuli Type'] == stimulus].reset_index(drop=True)
        avg_duration = stim_times['Duration'].mean()

        time_axis = np.arange(-padding, avg_duration + padding, dt)
        spikes = np.asarray(spikes)

        n_trials = len(stim_times)

        # ---- Vectorized Gaussian kernel sum ----
        # shape: (n_spikes, n_timepoints)
        diff = time_axis[None, :] - spikes[:, None]
        kernels = np.exp(-(diff ** 2) / (2 * sigma ** 2))

        rate = kernels.sum(axis=0)

        if normalize:
            rate /= (n_trials * sigma * np.sqrt(2 * np.pi))

        if plot:
            ax.plot(time_axis, rate, color=color)
            ax.set_ylabel("Response (Hz)")

        return ax, [time_axis, rate]

    def plot_temporal_pcc(self, stimulus, ax=None, padding=None, dt=0.01, window=50, step=1):
        import numpy as np

        if padding is None:
            padding = self.padding

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            ax=None,
            baseline=False,
            plot=False,
            color='blue',
            separate_trials=True,
            padding=padding
        )

        bins = np.arange(-padding, trial_duration + 2 * padding, dt)

        X = []
        for i, spikes in enumerate(trial_spikes):
            counts, _ = np.histogram(spikes, bins=bins)
            X.append(counts)
        X = np.array(X)
        Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        n_trials, T = Xz.shape
        out = []
        for t in range(0, T - window + 1, step):
            window_data = Xz[:, t:t + window]
            C = np.corrcoef(window_data)

            upper = C[np.triu_indices(n_trials, k=1)]
            out.append(np.nanmean(upper))
        out = np.array(out)

        time = bins[:-1]  # time for each bin center
        pcc_time = time[:len(out)]

        import matplotlib.pyplot as plt

        ax.plot(pcc_time, out, lw=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial-to-trial PCC")
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)

    def format_axes(self, axes, stimulus, baseline=False, padding=None):
        if padding is None:
            padding = self.padding
        stim_times = self.stimuli[self.stimuli['Stimuli Type'] == stimulus].reset_index(drop=True)
        duration = stim_times['Duration'].mean()

        ticks = [0, self.stimuli[self.stimuli['Stimuli Type'] == stimulus]['Duration'].mean()]
        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xlim(-padding, ticks[1] + padding)
            if ax != axes[-1]:
                ax.set_xticklabels([])  # hide labels for top subplots
            else:
                ax.set_xticklabels([f"{t:.1f}" for t in ticks])
        for ax in axes:
            ax.label_outer()  # only shows x labels for bottom axes

        return axes

    def save_plots(self, output_path=None, name_prefix='', format='png', clear_cache=True):
        import os
        import matplotlib.pyplot as plt

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            for stim in self.figures.keys():
                self.figures[stim].savefig(os.path.join(output_path, f'{name_prefix}{stim}.{format}'),
                                           bbox_inches='tight', format=format)
            if clear_cache:
                for fig in self.figures.values():
                    plt.close(fig)
                self.figures.clear()
        else:
            raise FileNotFoundError('No output path specified')

    def set_stimulus_data_for_beat_analysis(self, stimulus, psth_dt, psth_sigma):
        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            dt=psth_dt,
            sigma=psth_sigma,
            color='blue',
            plot=False
        )
        print("automatically setting trial data...")

        _, baseline_spikes, trial_duration = self.raster(
            stimulus,
            baseline=True,
            plot=False,
            color='red'
        )
        _, self.baseline_psth = self.psth(
            stimulus,
            baseline_spikes,
            normalize=True,
            dt=psth_dt,
            sigma=psth_sigma,
            color='red',
            plot=False
        )
        print("automatically setting baseline data...")

    def analyze_beats(self, stimulus, psth_dt=0.001, psth_sigma=0.015, save_path=None, save_format='.png',
                      padding=None):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if padding is None:
            padding = self.padding

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(
            3, 2,
            figure=fig,
            height_ratios=[1, 1, 2],
            hspace=0.5,
            wspace=0.3
        )
        self.set_stimulus_data_for_beat_analysis(stimulus, psth_dt, psth_sigma)

        stimulus_ax = fig.add_subplot(gs[0, 0])
        stimulus_ax.set_ylabel('Stimulus Amplitude (a.u.)', fontsize=8)
        _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, plot=True, ax=stimulus_ax)
        print("automatically setting stimulus data...")
        self.stimulus_time = self.stimulus_time - padding
        _, onsets, offsets = self.find_beat_times(ax=stimulus_ax)

        psth_ax = fig.add_subplot(gs[1, 0], sharex=stimulus_ax)
        adaptation_ax = fig.add_subplot(gs[2, :], sharex=stimulus_ax)
        segment_ax = fig.add_subplot(gs[0:2, 1])

        if len(onsets) == len(offsets):
            num_beats = len(onsets)
            print(f'There are {len(onsets)} beats.')
        _, _, adaptation, segment_psths, slope_trial, slope_base = self.find_adaptation(onsets, offsets, ax1=psth_ax,
                                                                                        ax2=adaptation_ax,
                                                                                        ax3=segment_ax, psth_dt=psth_dt,
                                                                                        num_beats=num_beats)

        plt.suptitle(f'Beat analysis of {self.unit_id} on {stimulus}')

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'unit_{self.unit_id}_{stimulus}_beat_analysis.{save_format}'),
                        format=save_format)
            print(f'Saved to {save_path}')
        plt.show()

        return slope_trial, slope_base

    def linear_regression(self, x, y, return_pred=False):
        """
        Perform simple linear regression (y = m*x + b)

        Parameters
        ----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        return_pred : bool
            If True, also return predicted y values

        Returns
        -------
        m : float
            Slope
        b : float
            Intercept
        y_pred : np.ndarray, optional
            Predicted y values (only if return_pred=True)
        """
        import numpy as np

        x = np.asarray(x)
        y = np.asarray(y)

        # Means
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Compute slope (m)
        m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        # Compute intercept (b)
        b = y_mean - m * x_mean

        if return_pred:
            y_pred = m * x + b
            return m, b, y_pred
        else:
            return m, b

    def find_adaptation(self, onsets, offsets, ax1=None, ax2=None, ax3=None, psth_dt=0.001, num_beats=10):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import linregress

        adaptation, segment_psths = {}, {}
        if 'trial' not in adaptation.keys():
            adaptation['trial'] = []
            segment_psths['trial'] = []
        if 'baseline' not in adaptation.keys():
            adaptation['baseline'] = []
            segment_psths['baseline'] = []

        if ax1 is None:
            fig1, ax1 = plt.subplots(1, 1, figsize=self.figure_size)
        if ax2 is None:
            fig2, ax2 = plt.subplots(1, 1, figsize=self.figure_size)
        if ax3 is None:
            fig3, ax3 = plt.subplots(1, 1, figsize=self.figure_size)

        time = self.trial_psth[0]
        trial_psth = self.trial_psth[1]
        baseline_psth = self.baseline_psth[1]
        psth_sr = 1 / psth_dt
        print(f'psth_sr = {psth_sr}')

        # calculate gap
        gaps, beats = [], []
        for e, edge in enumerate(onsets[1:]):
            gaps.append(edge - offsets[e - 1])
            beats.append(offsets[e + 1] - edge)
        gap = np.mean(gaps)
        beat = np.mean(beats)
        print(f'Average gap duration: {gap:.2f} seconds. Average beat duration: {beat:.2f} seconds.')

        for e, (onset, offset) in enumerate(zip(onsets, offsets)):
            start = onset - 0.02
            end = offset + gap - 0.01

            ax1.axvline(x=start, linestyle='--', color='green', alpha=0.4)
            ax1.axvline(x=end, linestyle='--', color='red', alpha=0.4)

            mask = (time <= end) & (time > start)
            segment_trial_psth = trial_psth[mask]
            segment_baseline_psth = baseline_psth[mask]
            adaptation['trial'].append(np.max(segment_trial_psth))
            adaptation['baseline'].append(np.max(segment_baseline_psth))
            segment_psths['trial'].append(segment_trial_psth)
            segment_psths['baseline'].append(segment_baseline_psth)

        ax1.plot(time, trial_psth, color='blue', alpha=0.2)
        ax1.plot(time, baseline_psth, color='red', alpha=0.2)
        ax1.set_ylabel('Firing Rate (Hz)', fontsize=8)
        ax1.set_xlabel('Time (s)', fontsize=8)

        ax2.scatter(onsets + beat / 2, adaptation['trial'], color='blue')
        ax2.scatter(onsets + beat / 2, adaptation['baseline'], color='red')
        ax2.set_ylabel('Maximum Segment Firing Rate (Hz)', fontsize=8)
        ax2.set_xlabel('Time (s)', fontsize=8)

        inc = 0.9 / num_beats
        for s, (trial_seg, baseline_seg) in enumerate(zip(segment_psths['trial'], segment_psths['baseline'])):
            ax3.plot(np.arange(len(trial_seg)) / psth_sr, trial_seg, color='blue', alpha=1 - inc * s)
            ax3.plot(np.arange(len(baseline_seg)) / psth_sr, baseline_seg, color='red', alpha=1 - 0.05 * s)
        ax3.set_ylabel('Firing Rate (Hz)', fontsize=8)
        ax3.set_xlabel('Time From Segment Start (s)', fontsize=8)

        # linear regression
        x = onsets + beat / 2

        # --- Regression lines (your function) ---
        slope_trial, intercept_trial, y_regress_trial = self.linear_regression(x, adaptation['trial'], return_pred=True)
        ax2.plot(x, y_regress_trial, color='blue', linestyle='--', alpha=0.6)

        slope_base, intercept_base, y_regress_base = self.linear_regression(x, adaptation['baseline'], return_pred=True)
        ax2.plot(x, y_regress_base, color='red', linestyle='--', alpha=0.6)

        # --- Statistical significance ---
        def stars(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return '(ns)'

        # Use linregress just for p-values
        reg_trial = linregress(x, adaptation['trial'])
        reg_base = linregress(x, adaptation['baseline'])

        star_trial = stars(reg_trial.pvalue)
        star_base = stars(reg_base.pvalue)

        # --- Legend with slope and significance ---
        ax2.legend([
            f"Trial: slope={slope_trial:.2f} {star_trial}",
            f"Baseline: slope={slope_base:.2f} {star_base}"
        ])

        return ax1, ax2, adaptation, segment_psths, slope_trial, slope_base

    def find_beat_times(self, ax=None, plot=False):
        import numpy as np
        import matplotlib.pyplot as plt

        if plot and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        # ====== DETECT BEATS =======
        audio_deriv = np.gradient(self.stimulus_data)

        from scipy.signal import find_peaks
        peaks_pos, _ = find_peaks(audio_deriv, height=0.05)  # rising edges
        peaks_neg, _ = find_peaks(-audio_deriv, height=0.05)  # falling edges
        all_edges = np.sort(np.concatenate((peaks_pos, peaks_neg)))

        edge_times = self.stimulus_time[all_edges]
        edge_times = np.sort(edge_times)

        grouped_edges = []
        current_group = [edge_times[0]]

        for time in edge_times[1:]:
            if time - current_group[-1] <= 0.005:  # 5 ms in seconds
                current_group.append(time)
            else:
                grouped_edges.append(current_group)
                current_group = [time]

        grouped_edges.append(current_group)

        rising_edges_times = []
        falling_edges_times = []
        for group in grouped_edges:
            start = group[0]
            stop = group[-1]

            if stop - start > 0.01:
                rising_edges_times.append(start)
                falling_edges_times.append(stop)

                if plot:
                    ax.axvline(x=start, linestyle='--', color='green', alpha=0.8)
                    ax.axvline(x=stop, linestyle='--', color='red', alpha=0.8)

        return ax, rising_edges_times, falling_edges_times

    def generate_video(self, stimulus, output_path=None, fps=30, psth_dt=0.001, psth_sigma=0.015, padding=0.5):
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from moviepy import VideoClip, AudioFileClip
        from moviepy.audio.AudioClip import AudioArrayClip
        import cv2
        import numpy as np

        fig = plt.figure(figsize=(8, 2 * 3))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5, wspace=0.3)

        ax_wave = fig.add_subplot(gs[0, 0])
        ax_wave.set_ylabel("Stimulus Amplitude (a.u.)")
        self.plot_stimulus(stimulus, ax=ax_wave, padding=padding)

        ax_raster = fig.add_subplot(gs[1, 0], sharex=ax_wave)
        ax_psth = fig.add_subplot(gs[2, 0], sharex=ax_wave)

        _, trial_spikes, avg_duration = self.raster(stimulus, ax=ax_raster, padding=padding)
        _, self.trial_psth = self.psth(stimulus, trial_spikes, ax=ax_psth, dt=psth_dt, sigma=psth_sigma,
                                       padding=padding)

        playhead_raster = ax_raster.axvline(-padding, color='red', lw=1, zorder=10)
        playhead_sound = ax_wave.axvline(-padding, color='red', lw=1, zorder=10)
        playhead_psth = ax_psth.axvline(-padding, color='red', lw=1, zorder=10)

        fps = 30
        fps_audio = 44100

        duration = avg_duration + padding * 2
        frames = int(duration * fps)

        audio_clip = AudioFileClip(os.path.join(self.db.stim_library, f'{stimulus}.wav'))
        y = audio_clip.to_soundarray(fps=44100)  # shape: (n_samples, nchannels)
        y = y / np.max(np.abs(y))  # ensures volume isn't near zero
        video_duration = avg_duration + 2 * padding
        n_samples_video = int(video_duration * fps_audio)
        padding_samples = int(padding * fps_audio)
        pad_front = np.zeros((padding_samples, y.shape[1]))
        pad_back = np.zeros((n_samples_video - y.shape[0] - padding_samples, y.shape[1]))

        y_padded = np.vstack([pad_front, y, pad_back])

        # create a MoviePy AudioClip from the padded array
        padded_audio = AudioArrayClip(y_padded, fps=fps_audio)

        def make_frame(t):
            """
            t: current time in seconds
            returns: RGB frame (H x W x 3)
            """
            # update playhead lines
            playhead_raster.set_xdata([t - padding, t - padding])
            playhead_sound.set_xdata([t - padding, t - padding])
            playhead_psth.set_xdata([t - padding, t - padding])

            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())  # H x W x 4
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # convert to 3 channels
            return img

        video_clip = VideoClip(make_frame, duration=duration)
        # video_clip = video_clip.set_fps(fps)
        # video_clip = video_clip.set_audio(audio_clip)

        video_clip.write_videofile(
            r"C:\Users\tmerri03\Desktop\Test Sorting\neuron_video_with_sound.mp4",
            codec="libx264",
            audio_codec="aac",
            audio=padded_audio,
            fps=fps
        )
        padded_audio.write_audiofile(r"C:\Users\tmerri03\Desktop\Test Sorting\stimulus.wav")

        print(padded_audio.fps, padded_audio.nchannels, padded_audio.duration)
        # def update(frame):
        #     t = -padding + frame / fps
        #
        #     playhead_raster.set_xdata([t, t])
        #     playhead_sound.set_xdata([t, t])
        #     playhead_psth.set_xdata([t, t])
        #
        #     return playhead_raster, playhead_sound, playhead_psth
        #
        # save_dir = r"C:\Users\tmerri03\Desktop\Test Sorting\frames"
        #
        # width, height = fig.canvas.get_width_height()
        #
        # video = cv2.VideoWriter(
        #     os.path.join(save_dir, "neuron_video.mp4"),
        #     cv2.VideoWriter_fourcc(*"mp4v"),
        #     fps,
        #     (width, height)
        # )
        #
        # for frame in range(frames):
        #     update(frame)  # your matplotlib update
        #
        #     fig.canvas.draw()
        #
        #     img = np.asarray(fig.canvas.buffer_rgba())
        #
        #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        #     video.write(img)
        #
        # video.release()

    def generate_video_test(self, stimulus, output_path=None, fps=30, psth_dt=0.001, psth_sigma=0.015, padding=0.5):
        from moviepy import VideoClip, AudioFileClip
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        fig = plt.figure(figsize=(8, 2 * 3))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5, wspace=0.3)

        ax_wave = fig.add_subplot(gs[0, 0])
        ax_wave.set_ylabel("Stimulus Amplitude (a.u.)")
        self.plot_stimulus(stimulus, ax=ax_wave, padding=padding)

        ax_raster = fig.add_subplot(gs[1, 0], sharex=ax_wave)
        ax_psth = fig.add_subplot(gs[2, 0], sharex=ax_wave)

        _, trial_spikes, avg_duration = self.raster(stimulus, ax=ax_raster, padding=padding)
        _, self.trial_psth = self.psth(stimulus, trial_spikes, ax=ax_psth, dt=psth_dt, sigma=psth_sigma,
                                       padding=padding)

        playhead_raster = ax_raster.axvline(-padding, color='red', lw=1, zorder=10)
        playhead_sound = ax_wave.axvline(-padding, color='red', lw=1, zorder=10)
        playhead_psth = ax_psth.axvline(-padding, color='red', lw=1, zorder=10)

        # --- Step 1: Prepare padded audio ---
        audio_clip = AudioFileClip(os.path.join(self.db.stim_library, f'{stimulus}.wav'))
        fps_audio = audio_clip.fps
        y = audio_clip.to_soundarray(fps=fps_audio)
        y = y / np.max(np.abs(y))  # normalize

        padding_samples = int(padding * fps_audio)
        n_samples_video = int((avg_duration + 2 * padding) * fps_audio)
        pad_front = np.zeros((padding_samples, y.shape[1]))
        pad_back = np.zeros((n_samples_video - y.shape[0] - padding_samples, y.shape[1]))
        y_padded = np.vstack([pad_front, y, pad_back])

        # Save as WAV for MoviePy to read reliably
        from scipy.io.wavfile import write
        write(r'C:\Users\tmerri03\Desktop\Test Sorting\stimulus_padded.wav', fps_audio,
              (y_padded * 32767).astype(np.int16))

        # --- Step 2: Create video ---
        def make_frame(t):
            playhead_raster.set_xdata([t - padding, t - padding])
            playhead_sound.set_xdata([t - padding, t - padding])
            playhead_psth.set_xdata([t - padding, t - padding])
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return img

        video_clip = VideoClip(make_frame, duration=avg_duration + 2 * padding)

        # Attach WAV audio using AudioFileClip
        audio_clip_padded = AudioFileClip(r"C:\Users\tmerri03\Desktop\Test Sorting\stimulus_padded.wav")

        from pydub import AudioSegment

        # Load your padded WAV audio
        audio = AudioSegment.from_wav(r"C:\Users\tmerri03\Desktop\Test Sorting\stimulus_padded.wav")

        # Export to MP4-compatible AAC audio track
        audio.export(r"C:\Users\tmerri03\Desktop\Test Sorting\stimulus_padded.m4a",
                     format="mp4")  # produces a .m4a audio file

        from moviepy.editor import VideoFileClip, AudioFileClip

        video_clip = VideoFileClip(r"C:\Users\tmerri03\Desktop\Test Sorting\neuron_video_video_only.mp4")
        audio_clip = AudioFileClip(r"C:\Users\tmerri03\Desktop\Test Sorting\stimulus_padded.m4a")

        video_clip_with_audio = video_clip.set_audio(audio_clip)
        video_clip_with_audio.write_videofile(
            "neuron_video_with_sound.mp4",
            codec="libx264",
            audio_codec="aac",
            fps=30
        )

    @staticmethod
    def phase_rose(phases):
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        ax.hist(phases, bins=30)

        mean_phase = np.angle(np.mean(np.exp(1j * phases)))

        ax.plot([mean_phase, mean_phase], [0, ax.get_ylim()[1]],
                color='red', linewidth=3)

        ax.set_title("Spike Phase Relative to Beat")

        plt.show()

    @staticmethod
    def compute_spike_phases(spike_times, beat_onsets, beat_offsets):
        """
        Compute phase (0-1) of each spike relative to its beat cycle.

        Returns:
            beat_numbers: array of beat indices for each spike
            spike_phases: array of phase (0-1) for each spike
        """
        import numpy as np

        beat_numbers = []
        spike_phases = []

        for i in range(len(beat_onsets) - 1):
            start = beat_onsets[i]
            end = beat_onsets[i + 1]  # actual beat duration
            duration = end - start

            mask = (spike_times >= start) & (spike_times < end)
            spikes = spike_times[mask]

            for spike in spikes:
                phase = (spike - start) / duration
                beat_numbers.append(i)
                spike_phases.append(phase)

        return np.array(beat_numbers), np.array(spike_phases)

    @staticmethod
    def compute_latency_v2(spike_times, beat_onsets):
        import numpy as np

        latencies = []
        for i in range(len(beat_onsets) - 1):
            start = beat_onsets[i]
            end = beat_onsets[i + 1]

            spikes_in_beat = spike_times[(spike_times >= start) & (spike_times < end)]
            latencies.extend(spikes_in_beat - start)
        mean_latency = np.mean(latencies)

        return mean_latency

    def predict_omission_response(self, stimulus, padding=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import gridspec

        if padding is None:
            padding = self.padding

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            height_ratios=[2, 3],
            width_ratios=[2, 1],
            hspace=0.2,
            wspace=0.1
        )
        phase_v_beats_ax = fig.add_subplot(gs[0, 0])
        cluster_ax = fig.add_subplot(gs[0, 1], polar=True)
        regular_stim = f'{stimulus.strip(" Omit")} REG'
        _, _, mean_latency, vector_strength, _, _ = self.calc_latency_v2(regular_stim, ax1=phase_v_beats_ax,
                                                                         ax2=cluster_ax)
        # mean_latency = self.calc_latency(regular_stim)

        prediction_ax = fig.add_subplot(gs[1, :])
        _, stim_time, self.stimulus_data = self.plot_stimulus(stimulus, plot=False)
        _, onsets, offsets = self.find_beat_times(plot=False)

        onsets = np.array(onsets)

        # 1. Find the beat before the omission
        iois = np.diff(onsets)  # inter-onset intervals
        omission_beat = np.argmax(iois)  # index of beat before long gap

        # 2. Compute average normal gap (excluding the omission)
        normal_iois = np.delete(iois, omission_beat)  # remove the long interval
        avg_gap = np.mean(normal_iois)  # average gap of regular beats

        print("Omission after beat:", omission_beat)
        print("Average normal gap:", avg_gap)

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            plot=False
        )
        prediction_ax.plot(self.trial_psth[0], self.trial_psth[1], color='blue', alpha=0.8)
        prediction_ax.plot(stim_time - padding, self.stimulus_data * max(self.trial_psth[1]), color='grey', alpha=0.5)

        for o in onsets:
            prediction_ax.axvline(x=o + mean_latency - padding, linestyle='--', color='green')
        prediction_ax.axvline(x=onsets[omission_beat] + avg_gap + mean_latency - padding, linestyle='--', color='red')

        print(f'mean latency: {mean_latency}')

        # prediction_ax.axvline(x=)
        plt.show()

    def calc_latency_v2(self, stimulus, ax1=None, ax2=None, plot=True):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.signal import correlate
        from scipy.signal import hilbert
        from scipy.signal import find_peaks

        _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, ax=None, plot=False)
        _, onsets, offsets = self.find_beat_times(plot=False)
        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue',
        )
        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            plot=False
        )
        ax1.plot(self.trial_psth[0], self.trial_psth[1], color='blue', alpha=0.8)

        # detect peaks
        peak_idx, properties = find_peaks(self.trial_psth[1], prominence=20)

        # convert to times
        peak_times = self.trial_psth[0][peak_idx]
        peak_heights = self.trial_psth[1][peak_idx]

        phases = []
        first_peak_times = []
        first_peak_heights = []

        beat_duration = np.mean(np.diff(onsets))

        for o in onsets:
            o = o - self.padding
            valid = peak_times[peak_times > o]
            if len(valid) == 0:
                continue

            first_peak = valid[0]

            # store for plotting
            first_peak_times.append(first_peak)
            first_peak_heights.append(
                peak_heights[np.where(peak_times == first_peak)][0]
            )

            latency = first_peak - o
            phase = latency / beat_duration
            phases.append(phase)

        # plot only the first peaks
        ax1.scatter(first_peak_times, first_peak_heights, color='red', s=60, zorder=3)

        phases = np.array(phases) * 2 * np.pi
        mean_phase = np.angle(np.mean(np.exp(1j * phases)))

        r = 1 + np.random.uniform(-0.05, 0.05, len(phases))
        ax2.scatter(phases, r, color='black', s=40)
        ax2.plot([mean_phase, mean_phase], [0, 1.1], color='red', linewidth=3)
        ax2.set_theta_zero_location("N")  # beat onset at top
        ax2.set_theta_direction(-1)  # clockwise
        ax2.set_yticks([])

        phases = np.array(phases) * 2 * np.pi  # convert to radians
        vector_strength = np.abs(np.mean(np.exp(1j * phases)))
        mean_latency = (mean_phase / (2 * np.pi)) * beat_duration

        return ax1, ax2, mean_latency, vector_strength, onsets, offsets

    def calc_latency(self, stimulus, ax1=None, ax2=None, plot=True):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.signal import correlate
        from scipy.signal import hilbert
        from scipy.signal import find_peaks

        _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, ax=None, plot=False)
        _, onsets, offsets = self.find_beat_times(plot=False)

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue',
        )
        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            plot=False
        )
        # ax1.plot(self.trial_psth[0], self.trial_psth[1], color='blue', alpha=0.8)
        find_peaks(self.trial_psth[1], height=5, prominence=5, threshold=0.5)
        peaks, properties = find_peaks(self.trial_psth[1], prominence=10)

        # # get peak coordinates
        peak_x = self.trial_psth[0][peaks]
        # peak_y = self.trial_psth[1][peaks]
        #
        # # plot peak markers
        # ax1.scatter(peak_x, peak_y, color='red', s=50, zorder=3)

        # variables
        # onsets, offsets: onset & offset times for each beat
        # trial_spikes: spikes for each trial concatenated
        # peaks: location of peaks of psth

        # trial_spikes = np.array(trial_spikes)
        peaks = np.array(peak_x)
        # mean_latency = self.compute_latency_v2(trial_spikes, onsets)
        mean_latency = self.compute_latency_v2(peaks, onsets)

        # beat_numbers, spike_phases = self.compute_spike_phases(trial_spikes, onsets, offsets)
        beat_numbers, spike_phases = self.compute_spike_phases(peaks, onsets, offsets)

        if plot:
            ax1.scatter(beat_numbers, spike_phases, s=15, alpha=0.6, color='blue')
            ax1.set_xlabel("Beat number")
            ax1.set_ylabel("Spike phase (0-1)")
            ax1.set_title("Spike Phase vs Beat Number")
            ax1.set_ylim(0, 1)

        spike_angles = spike_phases * 2 * np.pi
        _, mean_angle = self.plot_circular_cluster(spike_angles, ax=ax2, plot=plot)

        vector_strength = np.abs(np.mean(np.exp(1j * spike_angles)))
        mean_latency = (mean_angle / (2 * np.pi)) * np.mean(np.diff(onsets))
        return ax1, ax2, mean_latency, vector_strength, onsets, offsets

    @staticmethod
    def plot_circular_cluster(spike_angles, jitter_radius=1.0, point_color='blue', ax=None, plot=True):
        """
        Circular plot with spikes as points and optional radial jitter.

        Parameters:
            spike_angles : array of spike angles in radians
            jitter_radius : float, max radial jitter for visibility
            point_color : color of the spike points
        """
        import numpy as np
        import matplotlib.pyplot as plt
        # radial positions (random jitter for visibility)
        r = np.random.rand(len(spike_angles)) * jitter_radius

        # plot all spikes
        ax.scatter(spike_angles, r, s=20, alpha=0.7, color=point_color)

        # compute mean phase (circular mean)
        mean_angle = np.angle(np.mean(np.exp(1j * spike_angles)))
        if mean_angle < 0:
            mean_angle += 2 * np.pi

        # plot mean phase as a red line
        ax.plot([mean_angle, mean_angle], [0, jitter_radius], color='red', lw=2)

        ax.set_title("Circular Spike Cluster Plot")

        return ax, mean_angle

    @property
    def get_neuron_data(self):
        return (self.neuron_data)

    @property
    def get_manual_label(self):
        from pathlib import Path
        import json

        rec_name = Path(self.rec.rec_fp).name
        bird_id = rec_name.split(' ')[0]

        with open(self.manual_area_curation_path) as f:
            data = json.load(f)

        if rec_name in data.keys():
            if f'{self.unit_id}' in data[rec_name].keys():
                return data[rec_name][f'{self.unit_id}']

        else:
            return None

    def cross_correlation_analysis(self, stimulus, plot=True, correlate_with='onset', sigma=0.005):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.signal import correlate

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            dt=0.001,
            sigma=sigma,
            plot=False
        )

        _, baseline_spikes, baseline_duration = self.raster(
            stimulus,
            baseline=True,
            plot=False,
            color='red'
        )

        _, self.baseline_psth = self.psth(
            stimulus,
            baseline_spikes,
            normalize=True,
            color='red',
            dt=0.001,
            sigma=sigma,
            plot=False
        )

        norm_psth = [self.trial_psth[0], self.trial_psth[1] - self.baseline_psth[1]]

        _, time, sound = self.plot_stimulus(stimulus, plot=False, padding=None)
        self.stimulus_time = time
        self.stimulus_data = sound

        _, onsets, offsets = self.find_beat_times(plot=False)
        fs = 1 / 0.001
        if correlate_with == 'onset':
            onset_idx = (np.array(onsets) * fs).astype(int)
        elif correlate_with == 'offset':
            onset_idx = (np.array(offsets) * fs).astype(int)

        # stim=np.zeros_like(self.trial_psth[1])
        # stim[onset_idx] = 1
        #
        # diff = norm_psth[0][None, :] - stim[:, None]
        # kernels = np.exp(-(diff ** 2) / (2 * 0.005 ** 2))
        #
        # stim = kernels.sum(axis=0)

        onset_times = norm_psth[0][onset_idx]  # get the actual time of each onset
        diff = norm_psth[0][None, :] - onset_times[:, None]  # shape: n_onsets x n_time_bins
        kernels = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        stim_smooth = kernels.sum(axis=0)

        psth = norm_psth[1] - np.mean(norm_psth[1])
        stim = stim_smooth - np.mean(stim_smooth)

        corr = correlate(psth, stim, mode="full")
        lags = np.arange(-len(psth) + 1, len(psth))
        lags_sec = lags * 0.001

        norm_factor = np.sqrt(np.sum(psth ** 2) * np.sum(stim ** 2))
        corr_normalized = corr / norm_factor

        max_corr = np.max(corr_normalized)

        # max_lag = 2 * beat_interval
        # mask = (lags_sec >= 0) & (lags_sec <= max_lag)
        # corr = corr[mask]
        # lags_sec = lags_sec[mask]
        latency = lags_sec[np.argmax(corr_normalized)]
        # print("Estimated latency:", latency)

        if plot:
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8))

            axes[0].plot(self.trial_psth[0], self.trial_psth[1], color='blue')
            axes[0].plot(self.baseline_psth[0], self.baseline_psth[1], color='red')
            axes[0].plot(time - self.padding, abs(sound * max(self.trial_psth[1])), alpha=0.5, color='grey')
            axes[0].plot(self.trial_psth[0], stim * max(self.trial_psth[1]), color='green')

            axes[1].plot(norm_psth[0], norm_psth[1], color='blue')

            axes[2].plot(lags_sec, corr_normalized)
            axes[2].axvline(latency, linestyle="--")
            axes[2].set_xlabel("Lag (s)")
            axes[2].set_ylabel("Cross-correlation")

            axes[3].plot(norm_psth[0] - latency, norm_psth[1], color='blue')
            axes[3].plot(time - self.padding, abs(sound * max(self.trial_psth[1])), alpha=0.5, color='grey')
            axes[3].plot(self.trial_psth[0], stim * max(self.trial_psth[1]), color='green', alpha=0.5)

        return latency, max_corr

    def compute_latency_sd(self, stimulus, N_sd=5, plot=False, bins_over_threshold=30):
        import numpy as np
        import matplotlib.pyplot as plt

        _, t_spikes, _ = self.raster(stimulus, ax=None, plot=False, padding=0.5)
        _, t_psth_data = self.psth(stimulus, t_spikes, ax=None, plot=False, sigma=0.015, dt=0.001)
        _, b_spikes, _ = self.raster(stimulus, ax=None, plot=False, baseline=True, padding=0.5)
        _, b_psth_data = self.psth(stimulus, b_spikes, ax=None, plot=False, sigma=0.015, dt=0.001)
        t_psth_data = np.abs(t_psth_data)
        baseline_sd = np.std(b_psth_data[1])
        baseline_mean = np.mean(b_psth_data[1])

        threshold = baseline_mean + N_sd * baseline_sd
        signal = t_psth_data[1]
        time = t_psth_data[0]

        min_consecutive = bins_over_threshold  # <-- tune this

        crossings = signal > threshold

        # find runs of consecutive True values
        kernel = np.ones(min_consecutive, dtype=int)
        consecutive = np.convolve(crossings.astype(int), kernel, mode='valid') >= min_consecutive

        if not np.any(consecutive):
            latency = None
        else:
            idx = np.argmax(consecutive)
            _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, plot=False, ax=None)
            # onsets = self.find_beat_times(plot=True)
            # latency = time[idx]-onsets[0]
            latency = time[idx]
        if plot:
            plt.figure(figsize=(8, 4))

            # PSTHs
            plt.plot(t_psth_data[0], t_psth_data[1], color='blue', label='Trial PSTH')
            plt.plot(b_psth_data[0], b_psth_data[1], color='red', label='Baseline PSTH')

            # shaded 2x SD region (baseline mean ± 2 SD)
            plt.fill_between(
                b_psth_data[0],
                baseline_mean - N_sd * baseline_sd,
                baseline_mean + N_sd * baseline_sd,
                label='Baseline ±2 SD',
                color='red',
                alpha=0.15,
            )

            # threshold line (optional but useful)
            plt.axhline(baseline_mean + N_sd * baseline_sd, color='red', linestyle='--', alpha=0.7)
            plt.axhline(baseline_mean, color='red', linestyle='--', alpha=0.7)

            # latency line
            if latency is not None:
                plt.axvline(latency, color='black', linestyle='-', label=f'Latency = {latency:.3f}s')

            # labels
            plt.xlabel('Time (s)')
            plt.ylabel('Firing rate')
            plt.legend()
            plt.title('Trial vs Baseline PSTH with Latency Detection')

            plt.tight_layout()
            plt.show()

        return latency

    def diff_psth(self, stim1=None, stim2=None, plot=True, normalize_FRs=False):
        import matplotlib.pyplot as plt
        import numpy as np

        if stim1 is None or stim2 is None:
            raise ValueError("stim1 and stim2 are required")
            return None

        if plot:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
            stim1_ax = axes[0]
            stim2_ax = axes[1]
            diff_ax = axes[2]
        else:
            stim1_ax = None
            stim2_ax = None
            diff_ax = None

        # get psth1
        _, trial_spikes, trial_duration = self.raster(
            stim1,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, stim1_psth = self.psth(
            stim1,
            trial_spikes,
            normalize=True,
            color='blue',
            dt=0.001,
            sigma=0.015,
            plot=False,
            ax=stim1_ax
        )

        # get psth2
        _, trial_spikes, trial_duration = self.raster(
            stim2,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, stim2_psth = self.psth(
            stim2,
            trial_spikes,
            normalize=True,
            color='blue',
            dt=0.001,
            sigma=0.015,
            plot=False,
            ax=stim2_ax
        )

        time1 = stim1_psth[0]
        time2 = stim2_psth[0]
        psth1 = np.array(stim1_psth[1])
        psth2 = np.array(stim2_psth[1])

        if normalize_FRs:
            psth1 /= max(psth1)
            psth2 /= max(psth2)

        stim1_ax.plot(time1, psth1, color='blue')
        stim2_ax.plot(time2, psth2, color='red')

        stim2_interp = np.interp(time1, time2, psth2)

        diff = psth1 - stim2_interp

        if plot:
            diff_ax.plot(time1, np.abs(diff))
            diff_ax.fill_between(time1, np.abs(diff), 0, alpha=0.3, color='black')

        integral = np.trapezoid(np.abs(diff), time1)

        fig.suptitle(self.unit_id)

        return (integral)

    def analyze_phase(self, stimulus=None, plot=True, n_bins=50):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import numpy as np

        if stimulus is None:
            raise ValueError("stimulus is required")

        psth_ax = None
        trial_ax = None
        phase_ax = None

        if plot:
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(
                3, 2,
                figure=fig,
                height_ratios=[1, 1, 1],
                width_ratios=[2, 1],
                hspace=0.2,
                wspace=0.1
            )
            stimulus_ax = fig.add_subplot(gs[0, 0])
            psth_ax = fig.add_subplot(gs[2, 0])
            trial_ax = fig.add_subplot(gs[1, 0], sharex=psth_ax)
            onset_phase_ax = fig.add_subplot(gs[0, 1], polar=True)
            offset_phase_ax = fig.add_subplot(gs[2, 1], polar=True)

        _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, ax=stimulus_ax, plot=plot)

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=plot,
            color='blue',
            ax=trial_ax
        )
        _, stim_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            dt=0.001,
            sigma=0.015,
            plot=plot,
            ax=psth_ax
        )

        _, onsets, offsets = self.find_beat_times(plot=False)
        onsets = np.array(onsets) - self.padding
        offsets = np.array(offsets) - self.padding
        if plot:
            for onset in onsets:
                stimulus_ax.axvline(x=onset, color='green', linestyle='--', alpha=0.5)
                trial_ax.axvline(x=onset, color='green', linestyle='--', alpha=0.5)
                psth_ax.axvline(x=onset, color='green', linestyle='--', alpha=0.5)

            for offset in offsets:
                stimulus_ax.axvline(x=offset, color='red', linestyle='--', alpha=0.5)
                trial_ax.axvline(x=offset, color='red', linestyle='--', alpha=0.5)
                psth_ax.axvline(x=offset, color='red', linestyle='--', alpha=0.5)

        ioi = np.mean(np.diff(onsets)[:2])
        beat_duration = np.mean((np.array(offsets) - np.array(onsets))[:2])
        gap_duration = ioi - beat_duration

        print(f'-----------{stimulus}-----------')
        print(f'    ioi: {ioi * 1000:.2f}')
        print(f'    beat duration: {beat_duration * 1000:.2f}')
        print(f'    gap duration: {gap_duration * 1000:.2f}')
        print(f'')

        onset_line = 0
        offset_line = beat_duration / ioi * (2 * np.pi)
        _, onset_centers, onset_counts = self.plot_phases(onsets, trial_spikes, n_bins=n_bins, plot=True,
                                                          ax=onset_phase_ax, line1=onset_line, line2=offset_line,
                                                          title='Onset-aligned Phase')

        onset_line = gap_duration / ioi * (2 * np.pi)
        offset_line = 0
        _, offset_centers, offset_counts = self.plot_phases(offsets, trial_spikes, n_bins=n_bins, plot=True,
                                                            ax=offset_phase_ax, line1=onset_line, line2=offset_line,
                                                            title='Offset-aligned Phase')
        plt.show()

        return [onset_centers, onset_counts], [offset_centers, offset_counts]

    def plot_phases(self, baseline_time, trial_spikes, n_bins=50, plot=True, ax=None, line1=None, line2=None,
                    title='Phase Distribution'):
        import numpy as np
        import matplotlib.pyplot as plt

        trial_spikes = np.asarray(trial_spikes)
        onsets = np.asarray(baseline_time)

        # assign each spike to its interval
        idx = np.searchsorted(onsets, trial_spikes, side='right') - 1

        # keep only valid intervals
        valid = (idx >= 0) & (idx < len(onsets) - 1)

        spikes = trial_spikes[valid]
        idx = idx[valid]

        # interval boundaries
        t0 = onsets[idx]
        t1 = onsets[idx + 1]

        # relative time within interval
        rel = spikes - t0
        dur = t1 - t0

        # phase
        phases = 2 * np.pi * rel / dur

        counts, edges = np.histogram(phases, bins=n_bins, range=(0, 2 * np.pi), density=True)

        # bin centers
        centers = (edges[:-1] + edges[1:]) / 2

        # close the curve (important!)
        centers = np.append(centers, centers[0])
        counts = np.append(counts, counts[0])

        if plot and ax is not None:
            ax.plot(centers, counts, linewidth=2)
            ax.set_title(title)
            ax.set_yticklabels([])
            if line1 is not None:
                ax.plot([line1, line1], [0, max(counts)], color='green', linestyle='--')

            if line2 is not None:
                ax.plot([line2, line2], [0, max(counts)], color='red', linestyle='--')

            if line1 is not None and line2 is not None:
                rmax = ax.get_ylim()[1]

                if line1 < line2:
                    theta = np.linspace(line1, line2, 200)
                    ax.fill_between(theta, 0, rmax, alpha=0.2, color='grey')
                else:
                    theta1 = np.linspace(line1, 2 * np.pi, 100)
                    theta2 = np.linspace(0, line2, 100)

                    ax.fill_between(theta1, 0, rmax, alpha=0.2, color='grey')
                    ax.fill_between(theta2, 0, rmax, alpha=0.2, color='grey')
        return ax, centers, counts

    def _compare_omission_resonses(self, stim1, stim2):
        import numpy as np

        def circular_mean(theta, weights=None):
            if weights is None:
                weights = np.ones_like(theta)

            x = np.sum(weights * np.cos(theta))
            y = np.sum(weights * np.sin(theta))

            mean_angle = np.arctan2(y, x)
            r = np.sqrt(x ** 2 + y ** 2) / np.sum(weights)  # vector strength

            return mean_angle, r

        import matplotlib.pyplot as plt

        stim1_flower_onset, stim1_flower_offset = self.analyze_phase(stim1)
        stim2_flower_onset, stim2_flower_offset = self.analyze_phase(stim2)

        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

        import numpy as np

        theta1, r1 = circular_mean(stim1_flower_onset[0], stim1_flower_onset[1])
        theta2, r2 = circular_mean(stim2_flower_onset[0], stim2_flower_onset[1])

        axes[0].set_title('Aligned to Sound Onset')

        axes[0].plot(stim1_flower_onset[0], stim1_flower_onset[1], linewidth=2, color='blue')
        axes[0].plot(stim2_flower_onset[0], stim2_flower_onset[1], linewidth=2, color='red')

        # mean vectors
        axes[0].plot([theta1, theta1], [0, r1], color='blue', linewidth=3)
        axes[0].plot([theta2, theta2], [0, r2], color='red', linewidth=3)
        print(f'Onset mean vector difference: {abs(theta1 - theta2)}')

        theta1_off, r1_off = circular_mean(stim1_flower_offset[0], stim1_flower_offset[1])
        theta2_off, r2_off = circular_mean(stim2_flower_offset[0], stim2_flower_offset[1])

        axes[1].set_title('Aligned to Sound Offset')

        axes[1].plot(stim1_flower_offset[0], stim1_flower_offset[1], linewidth=2, color='blue')
        axes[1].plot(stim2_flower_offset[0], stim2_flower_offset[1], linewidth=2, color='red')

        # mean vectors
        axes[1].plot([theta1_off, theta1_off], [0, r1_off], color='blue', linewidth=3)
        axes[1].plot([theta2_off, theta2_off], [0, r2_off], color='red', linewidth=3)
        print(f'Offset mean vector difference: {abs(theta1_off - theta2_off)}')
        plt.show()

    def plot_waveform(self, ax=None, color1='black', color2='blue'):
        """
        Plot the mean waveform ± std onto a provided axis.
        (Modified from standalone version to accept an ax argument.)
        """
        from pathlib import Path
        import numpy as np

        if ax is None:
            fig, ax = plt.figure()

        spike_file = self.neuron_data['spike_file']
        waveforms_file = Path(spike_file) / '..' / 'waveforms.h5'

        waveform_data, standard_deviation, width_pp, width_hw = self.load_waveforms(waveforms_file, self.unit_id)
        x = np.arange(len(waveform_data))

        ax.plot(x, waveform_data, color=color1)
        ax.fill_between(
            x,
            waveform_data - standard_deviation,
            waveform_data + standard_deviation,
            alpha=0.2, color=color2
        )

        # --- TEXT: spike widths ---
        text_str = f"Peak-to-peak: {width_pp:.2f}ms\nHalf-width: {width_hw:.2f}ms"
        ax.text(
            0.98, 0.98, text_str,
            transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # --- SCALE BAR ---
        x_scale = 5  # samples
        y_scale = 50  # µV

        x0 = 0.05 * len(x)
        y_range = np.max(waveform_data) - np.min(waveform_data)
        y0 = np.min(waveform_data) - 0.05 * y_range  # relative offset, won't clip

        ax.plot([x0, x0 + x_scale], [y0, y0], color=color1, linewidth=2)
        ax.text(x0 + x_scale / 2, y0 - 0.03 * y_range,
                f"{x_scale / 30000 * 1000:.2f} ms", ha='center', va='top', fontsize=8)

        ax.plot([x0, x0], [y0, y0 + y_scale], color=color1, linewidth=2)
        ax.text(x0 - 0.5, y0 + y_scale / 2,
                f"{y_scale} µV", ha='right', va='center', rotation=90, fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def compute_pcc(self, stimulus):
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        _, trials, duration = self.raster(stimulus, ax=None, plot=False, separate_trials=True)

        t_min = -self.padding
        t_max = duration + self.padding
        bin_size = 0.001
        bins = np.arange(t_min, t_max + bin_size, bin_size)

        n_trials = len(trials)
        n_bins = len(bins) - 1
        binned_trials = np.zeros((n_trials, n_bins), dtype=float)

        for i, trial in enumerate(trials):
            binned_trials[i], _ = np.histogram(trial, bins=bins)
        binned_trials = gaussian_filter1d(binned_trials, sigma=5, axis=1)

        xcorr_matrix = np.corrcoef(binned_trials)
        xcorr_matrix = np.nan_to_num(xcorr_matrix)
        n = xcorr_matrix.shape[0]
        return np.mean(xcorr_matrix[np.triu_indices(n, k=1)])

    def compute_latency_shortened_stim(self, stimulus, plot=True, sigma=0.015, correlate_with='onset'):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import correlate

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            dt=0.001,
            sigma=sigma,
            plot=False
        )

        _, baseline_spikes, baseline_duration = self.raster(
            stimulus,
            baseline=True,
            plot=False,
            color='red'
        )

        _, self.baseline_psth = self.psth(
            stimulus,
            baseline_spikes,
            normalize=True,
            color='red',
            dt=0.001,
            sigma=sigma,
            plot=False
        )

        norm_psth = [self.trial_psth[0], self.trial_psth[1] - self.baseline_psth[1]]

        _, time, sound = self.plot_stimulus(stimulus, plot=False, padding=None)
        self.stimulus_time = time
        self.stimulus_data = sound
        _, onsets, offsets = self.find_beat_times(plot=False)

        ioi = np.mean(np.diff(onsets[:5]))
        threshold = ioi + ioi / 3
        diffs = np.diff(onsets)
        omission_idx = np.where(diffs > threshold)[0]
        omission_idx = omission_idx[0]
        pre_onset = onsets[omission_idx]

        fs = 1 / 0.001

        # -- correlate with omission stim --
        if correlate_with == 'onset':
            onset_idx = (np.array(onsets) * fs).astype(int)
        elif correlate_with == 'offset':
            onset_idx = (np.array(offsets) * fs).astype(int)
        onset_times = norm_psth[0][onset_idx]  # get the actual time of each onset
        diff = norm_psth[0][None, :] - onset_times[:, None]  # shape: n_onsets x n_time_bins
        kernels = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        stim_smooth = kernels.sum(axis=0)

        psth = norm_psth[1] - np.mean(norm_psth[1])
        stim = stim_smooth - np.mean(stim_smooth)
        time = self.trial_psth[0]

        mask = (time < 1)
        short_psth = psth[mask]
        short_stim = stim[mask]
        short_time = time[mask]

        corr = correlate(short_psth, short_stim, mode="full")
        lags = np.arange(-len(short_psth) + 1, len(short_psth))
        lags_sec = lags * 0.001

        norm_factor = np.sqrt(np.sum(short_psth ** 2) * np.sum(short_stim ** 2))
        corr_normalized = corr / norm_factor

        max_corr = np.max(corr_normalized)
        latency = lags_sec[np.argmax(corr_normalized)]

        if plot:
            fig = plt.figure()
            gs = fig.add_gridspec(3, 1)  # 3x3 grid
            corr_ax = fig.add_subplot(gs[0])  # Top row span

            corr_ax.plot(short_time, short_stim * max(psth))
            corr_ax.plot(short_time, short_psth, linestyle='--')
            corr_ax.plot(short_time - latency, short_psth, label=f'Latency: {latency}\nMax Corr: {max_corr:2f}')
            corr_ax.legend()

        return latency, max_corr

    def compute_omission_rs(self, stimulus, plot=True):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks

        fig = plt.figure()
        gs = fig.add_gridspec(3, 1)  # 3x3 grid
        stim_ax = fig.add_subplot(gs[0])  # Top row span
        psth_ax = fig.add_subplot(gs[1], sharex=stim_ax)
        # corr_ax = fig.add_subplot(gs[2], sharex=psth_ax)

        _, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stimulus, plot=True, ax=stim_ax)
        _, onsets, offsets = self.find_beat_times(plot=False)

        offsets = np.array(offsets) - self.padding
        onsets = np.array(onsets) - self.padding
        duration = np.mean(offsets[:5] - onsets[:5])

        ioi = np.mean(np.diff(offsets[:5]))
        diffs = np.diff(offsets)
        normal_beat_onsets = onsets

        # Beat BEFORE omission
        omission_beat_onsets = []
        threshold = ioi + ioi / 3
        omission_idx = np.where(diffs > threshold)[0]
        omission_idx = omission_idx[0]
        pre_onset = onsets[omission_idx]
        pre_offset = offsets[omission_idx]
        omission_beat_onsets.append(pre_onset + ioi)

        # Beat BEFORE sequence
        pre_stim_beat_onsets = []
        first_onset = normal_beat_onsets[0]
        pre_stim_beat_onsets.append(first_onset - ioi)

        # Beat AFTER sequence
        post_stim_beat_onsets = []
        last_onset = normal_beat_onsets[-1]
        post_stim_beat_onsets.append(last_onset + ioi)

        _, trial_spikes, trial_duration = self.raster(
            stimulus,
            baseline=False,
            plot=False,
            color='blue'
        )
        _, baseline_spikes, baseline_duration = self.raster(
            stimulus,
            baseline=True,
            plot=False,
            color='red'
        )

        _, self.trial_psth = self.psth(
            stimulus,
            trial_spikes,
            normalize=True,
            color='blue',
            plot=False,
            ax=None,
            sigma=0.015,
            dt=0.001
        )
        _, self.baseline_psth = self.psth(
            stimulus,
            baseline_spikes,
            normalize=True,
            color='red',
            plot=False,
            ax=None,
            sigma=0.015,
            dt=0.001
        )

        response_strength = self.trial_psth[1] - self.baseline_psth[1]
        rs_time = self.trial_psth[0]
        latency, max_corr = self.compute_latency_shortened_stim(stimulus, plot=False, correlate_with='offset')
        psth_ax.plot(rs_time, response_strength, linestyle='--')
        psth_ax.plot(rs_time - latency, response_strength)

        # corr_ax.plot(rs_time, response_strength, linestyle='--')
        # corr_ax.plot(rs_time-latency, response_strength)
        # print(max_corr)

        for o in normal_beat_onsets:
            stim_ax.hlines(xmin=o, xmax=o + duration, color='green', y=-1)
            psth_ax.hlines(xmin=o, xmax=o + duration, color='green', y=min(response_strength - 0.1))
            # corr_ax.hlines(xmin=o, xmax=o + duration, color='green', y=min(response_strength - 0.1))
        for o in omission_beat_onsets:
            stim_ax.hlines(xmin=o, xmax=o + duration, color='red', y=-1)
            psth_ax.hlines(xmin=o, xmax=o + duration, color='red', y=min(response_strength - 0.1))
            # corr_ax.hlines(xmin=o, xmax=o + duration, color='red', y=min(response_strength - 0.1))
        for o in pre_stim_beat_onsets:
            stim_ax.hlines(xmin=o, xmax=o + duration, color='yellow', y=-1)
            psth_ax.hlines(xmin=o, xmax=o + duration, color='yellow', y=min(response_strength - 0.1))
            # corr_ax.hlines(xmin=o, xmax=o+duration, color='red', y=min(response_strength-0.1))
        for o in post_stim_beat_onsets:
            stim_ax.hlines(xmin=o, xmax=o + duration, color='red', y=-1)
            psth_ax.hlines(xmin=o, xmax=o + duration, color='red', y=min(response_strength - 0.1))
            # corr_ax.hlines(xmin=o, xmax=o+duration, color='red', y=min(response_strength-0.1))

        phase = (rs_time % ioi) / ioi

        n_phase_bins = 50

        # Shift time by latency
        rs_time_shifted = rs_time - latency

        # Shift by latency
        rs_time_shifted = rs_time - latency
        phase_axis = np.linspace(0, 2 * np.pi, n_phase_bins)

        # Helper function to fold each beat into phase
        def fold_beats(beat_onsets, color, alpha=0.3, linewidth=1.5):
            for b in beat_onsets:
                mask = (rs_time_shifted >= b) & (rs_time_shifted < b + ioi)
                if np.any(mask):
                    phase = (rs_time_shifted[mask] - b) / ioi * 2 * np.pi
                    rs_bin = response_strength[mask]

                    # bin into n_phase_bins for smoother curve
                    binned = np.zeros(n_phase_bins)
                    bin_edges = np.linspace(0, 2 * np.pi, n_phase_bins + 1)
                    for j in range(n_phase_bins):
                        idx = (phase >= bin_edges[j]) & (phase < bin_edges[j + 1])
                        if np.any(idx):
                            binned[j] = rs_bin[idx].mean()
                    plt.polar(phase_axis, binned, color=color, alpha=alpha, linewidth=linewidth)

        self.figures[f'latency_{stimulus}'] = fig

        # plt.figure(figsize=(6, 6))
        #
        # # Plot each type of beat
        # fold_beats(pre_stim_beat_onsets, color='yellow')
        # fold_beats(normal_beat_onsets, color='green')
        # fold_beats(omission_beat_onsets, color='red')
        # fold_beats(post_stim_beat_onsets, color='blue')
        #
        # # Formatting
        # ax = plt.gca()
        # ax.set_theta_zero_location("N")  # phase=0 at top
        # ax.set_theta_direction(1)  # clockwise
        # ax.set_xticks(np.linspace(0, 2 * np.pi, 8))
        # ax.set_xticklabels(['0', '1/8', '1/4', '3/8', '1/2', '5/8', '3/4', '7/8'])
        # ax.set_yticklabels([])
        # plt.title('Beat-by-beat phase-PSTH (latency-corrected)')
        #
        # # Legend: plot dummy lines
        # for c, label in zip(['yellow', 'green', 'red', 'blue'],
        #                     ['Pre-stim', 'Stim', 'Omission', 'Post-stim']):
        #     plt.plot([], [], color=c, label=label, linewidth=2)
        # plt.legend(loc='upper right')

    # def compare_cc(self, stimulus, sigma=0.015):
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     from scipy.signal import correlate
    #
    #     fig = plt.figure()
    #     gs = fig.add_gridspec(3, 1)  # 3x3 grid
    #     psth_ax = fig.add_subplot(gs[1])
    #     corr_ax = fig.add_subplot(gs[0], sharex=psth_ax)  # Top row span
    #
    #     _, trial_spikes, trial_duration = self.raster(
    #         stimulus,
    #         baseline=False,
    #         plot=False,
    #         color='blue'
    #     )
    #     _, self.trial_psth = self.psth(
    #         stimulus,
    #         trial_spikes,
    #         normalize=True,
    #         color='blue',
    #         dt=0.001,
    #         sigma=sigma,
    #         plot=False
    #     )
    #
    #     _, baseline_spikes, baseline_duration = self.raster(
    #         stimulus,
    #         baseline=True,
    #         plot=False,
    #         color='red'
    #     )
    #
    #     _, self.baseline_psth = self.psth(
    #         stimulus,
    #         baseline_spikes,
    #         normalize=True,
    #         color='red',
    #         dt=0.001,
    #         sigma=sigma,
    #         plot=False
    #     )
    #
    #     norm_psth = [self.trial_psth[0], self.trial_psth[1] - self.baseline_psth[1]]
    #
    #     _, time, sound = self.plot_stimulus(stimulus, plot=False, padding=None)
    #     self.stimulus_time = time
    #     self.stimulus_data = sound
    #     _, onsets, offsets = self.find_beat_times(plot=False)
    #
    #     ioi = np.mean(np.diff(onsets[:5]))
    #     threshold = ioi + ioi / 3
    #     diffs = np.diff(onsets)
    #     omission_idx = np.where(diffs > threshold)[0]
    #     omission_idx = omission_idx[0]
    #     pre_onset = onsets[omission_idx]
    #
    #     fs = 1 / 0.001
    #
    #     # -- correlate with omission stim --
    #     onset_idx = (np.array(onsets) * fs).astype(int)
    #     onset_times = norm_psth[0][onset_idx]  # get the actual time of each onset
    #     diff = norm_psth[0][None, :] - onset_times[:, None]  # shape: n_onsets x n_time_bins
    #     kernels = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    #     stim_smooth = kernels.sum(axis=0)
    #
    #     psth = norm_psth[1] - np.mean(norm_psth[1])
    #     stim = stim_smooth - np.mean(stim_smooth)
    #     time = self.trial_psth[0]
    #
    #     mask = (time < 1)
    #     short_psth = psth[mask]
    #     short_stim = stim[mask]
    #     short_time = time[mask]
    #
    #     corr = correlate(short_psth, short_stim, mode="full")
    #     lags = np.arange(-len(short_psth) + 1, len(short_psth))
    #     lags_sec = lags * 0.001
    #
    #     norm_factor = np.sqrt(np.sum(short_psth ** 2) * np.sum(short_stim ** 2))
    #     corr_normalized = corr / norm_factor
    #
    #
    #     max_corr = np.max(corr_normalized)
    #     latency = lags_sec[np.argmax(corr_normalized)]
    #     corr_ax.plot(short_time, short_stim*max(psth))
    #     corr_ax.plot(short_time, short_psth, linestyle='--')
    #     corr_ax.plot(short_time-latency, short_psth, label=f'Latency: {latency}\nMax Corr: {max_corr:2f}')
    #     corr_ax.legend()
    #
    #     psth_ax.plot(time, stim*max(psth))
    #     psth_ax.plot(time, psth, linestyle='--')
    #     psth_ax.plot(time-latency, psth)
    #
    #     # # -- correlate with regular stim --
    #     # onsets = sorted(np.append(onsets, pre_onset + ioi))
    #     # onset_idx = (np.array(onsets) * fs).astype(int)
    #     # onset_times = norm_psth[0][onset_idx]  # get the actual time of each onset
    #     # diff = norm_psth[0][None, :] - onset_times[:, None]  # shape: n_onsets x n_time_bins
    #     # kernels = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    #     # stim_smooth = kernels.sum(axis=0)
    #     #
    #     # psth = norm_psth[1] - np.mean(norm_psth[1])
    #     # stim = stim_smooth - np.mean(stim_smooth)
    #     # mask = (time<1)
    #     # psth = psth[mask]
    #     # stim = stim[mask]
    #     # time = time[mask]
    #     #
    #     # corr = correlate(psth, stim, mode="full")
    #     # lags = np.arange(-len(psth) + 1, len(psth))
    #     # lags_sec = lags * 0.001
    #     #
    #     # norm_factor = np.sqrt(np.sum(psth ** 2) * np.sum(stim ** 2))
    #     # corr_normalized = corr / norm_factor
    #     #
    #     # max_corr = np.max(corr_normalized)
    #     # latency = lags_sec[np.argmax(corr_normalized)]
    #     # reg_ax.plot(time, stim * max(psth))
    #     # reg_ax.plot(time, psth, linestyle='--')
    #     # reg_ax.plot(time - latency, psth, label=f'Latency: {latency}\nMax Corr: {max_corr:2f}')
    #     # reg_ax.legend()
    #     #
    #     # latency, max_corr = self.cross_correlation_analysis("White Noise", plot=False)
    #     # wn_ax.plot(time, stim * max(psth))
    #     # wn_ax.plot(time, psth, linestyle='--')
    #     # wn_ax.plot(time - latency, psth, label=f'Latency: {latency}\nMax Corr: {max_corr:2f}')
    #     # wn_ax.legend()

    def calculate_rs(self, stim, random_permutation=False):
        import matplotlib.pyplot as plt
        import numpy as np

        _, trial_spikes, trial_duration = self.raster(
            stim,
            ax=None,
            baseline=False,
            plot=False,
            color='blue',
            padding=0
        )
        _, baseline_spikes, _ = self.raster(
            stim,
            ax=None,
            baseline=True,
            plot=False,
            color='red'
        )
        if random_permutation:
            trial_spikes = np.random.permutation(trial_spikes)
            baseline_spikes = np.random.permutation(baseline_spikes)

        rs = (len(trial_spikes) - len(baseline_spikes)) / trial_duration

        _, self.trial_psth = self.psth(
            stim,
            trial_spikes,
            ax=None,
            normalize=True,
            dt=0.001,
            sigma=0.015,
            color='blue',
            plot=False
        )
        _, self.baseline_psth = self.psth(
            stim,
            baseline_spikes,
            ax=None,
            normalize=True,
            dt=0.001,
            sigma=0.015,
            color='red',
            plot=False
        )
        instantaneous_rs = np.array(self.trial_psth) - np.array(self.baseline_psth)
        max_rs = np.max(instantaneous_rs)
        min_rs = np.min(instantaneous_rs)
        if abs(min_rs) > max_rs:
            max_rs = min_rs

        return rs, max_rs

    @property
    def is_responsive(self):
        import pandas as pd
        import numpy as np

        rhythm_stim = ['120', '144', '180', '220']
        stim = pd.unique(self.stimuli['Stimuli Type'])
        valid_stim = [s for s in stim if any(r in s for r in rhythm_stim)]
        for vs in valid_stim:
            self.reason = None

            _, t_spikes, _ = self.raster(vs, ax=None, plot=False, padding=0)
            _, t_psth_data = self.psth(vs, t_spikes, ax=None, plot=False, sigma=0.015, dt=0.001)
            _, b_spikes, _ = self.raster(vs, ax=None, plot=False, baseline=True, padding=0)
            _, b_psth_data = self.psth(vs, b_spikes, ax=None, plot=False, sigma=0.015, dt=0.001)

            baseline_sd = np.std(b_psth_data[1])
            baseline_mean = np.mean(b_psth_data[1])
            max_rs = abs(np.max(np.array(t_psth_data[1])))
            if max_rs > baseline_mean + 2 * baseline_sd:
                if int(abs(np.max(b_psth_data[1]))) >= int(max_rs):
                    self.reason = f'Max Trial PSTH < Max Baseline PSTH'
                    continue

                _, t_spikes, _ = self.raster(vs, ax=None, plot=False, padding=0, separate_trials=True)
                num_trials = len(t_spikes)
                r = 0
                for t in t_spikes:
                    if len(t) > 0:
                        r += 1
                if r / num_trials * 100 < 70:
                    self.reason = f'Only spikes in {r / num_trials * 100} percent of trials.'
                    continue

                # print('-------------')
                # print(f'baseline_mean: {baseline_mean}')
                # print(f'baseline_sd: {baseline_sd}')
                # print()
                # print(f'max_fr: {max_rs}')
                # print(f'max_bl: {abs(np.max(b_psth_data[1]))}')
                # print('-------------')
                #
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1,1)
                # ax.plot(t_psth_data[0], np.array(t_psth_data[1]), color='blue')
                # ax.axhline(y=baseline_mean, color='red', linestyle='--')
                # ax.fill_between(t_psth_data[0],y1=baseline_mean-2*baseline_sd, y2=baseline_mean+2*baseline_sd, color='red')
                # plt.show()

                # self.plot(baseline=True, padding=0, stimuli_to_raster=[vs])

                return True
        return False

    @property
    def peak_channel(self):
        import numpy as np
        import os
        from open_ephys.analysis import Session
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        rec_path = self.rec.rec_fp
        filtered_path = os.path.join(rec_path, 'filtered')
        oe_folder = [
            os.path.join(filtered_path, f)
            for f in os.listdir(filtered_path)
            if os.path.isdir(os.path.join(filtered_path, f))
        ][0]

        session = Session(oe_folder)
        recording = session.recordnodes[0].recordings[0]

        kilosort_path = os.path.join(self.rec.rec_fp, 'sorting', 'sorting_TDC', 'sorter_output')
        spike_clusters = np.load(kilosort_path + "/spike_clusters.npy")
        spike_templates = np.load(kilosort_path + "/spike_templates.npy")
        templates = np.load(kilosort_path + "/templates.npy")
        channel_map = np.load(kilosort_path + "/channel_map.npy")
        # (1) Find template ID associated with this cluster
        tmpl_ids = spike_templates[spike_clusters == self.unit_id]
        if len(tmpl_ids) == 0:
            raise ValueError(f"No spikes found for cluster {self.unit_id}")

        best_template = np.bincount(tmpl_ids).argmax()

        # (2) Extract the template (shape: timepoints × channels)
        template = templates[best_template]  # (T, C)

        # (3) Compute magnitude per channel (same metric Phy uses)
        # peak absolute amplitude across time for each channel
        peak_abs = np.max(np.abs(template), axis=0)  # (C,)

        # (4) Find channel with the largest amplitude in template space
        peak_raw = np.argmax(peak_abs)  # index 0..C-1

        # (5) Convert template-index → physical channel number (Phy index)
        channel = int(channel_map[peak_raw])
        return channel

    def plot_raw_data(self, stim, ax=None):
        import numpy as np
        import os
        from open_ephys.analysis import Session
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        channel = self.peak_channel

        this_stimulus_times = self.stimuli[self.stimuli['Stimuli Type'] == stim].reset_index(drop=True).loc[5]
        stim_pre = int((this_stimulus_times['Start Time'] - self.padding) * self.rec.samplerate)
        stim_post = int((this_stimulus_times['End Time'] + self.padding) * self.rec.samplerate)

        oe_path = os.path.join(self.rec.rec_fp, 'filtered')
        subdir = [d for d in os.listdir(oe_path) if os.path.isdir(os.path.join(oe_path, d))][0]
        oe_path = os.path.join(oe_path, subdir)

        session = Session(oe_path)
        recording = session.recordnodes[0].recordings[0]
        channel_data = recording.continuous[0].get_samples(start_sample_index=stim_pre, end_sample_index=stim_post,
                                                           selected_channels=[channel])
        if ax is None:
            fig = plt.figure(figsize=(8, 2 * 1))
            gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.5, wspace=0.3)
            ax = fig.add_subplot(gs[0, 0])
        n_samples = len(channel_data)
        time = np.arange(n_samples) / self.rec.samplerate
        time = time - self.padding
        ax.plot(time, channel_data, linewidth=0.2, color='grey')
        ax.margins(x=0)
        ax.set_ylabel('Voltage (uV)')
        ax.set_yticks([-100, 0, 100])

        return ax, channel_data

    def _irregular_response(self, stimulus, padding=0.5, psth_dt=0.001, psth_sigma=0.015, show=True):
        import numpy as np
        import matplotlib.pyplot as plt

        # Build the standard multi-panel response figure first.
        self.plot(
            raster=True,
            psth=True,
            baseline=True,
            stimuli_to_raster=[stimulus],
            padding=padding,
            psth_dt=psth_dt,
            psth_sigma=psth_sigma,
            show=False
        )

        fig = self.figures.get(stimulus)
        if fig is None or len(fig.axes) < 4:
            raise RuntimeError(f"Could not find plotting axes for stimulus '{stimulus}'.")

        # Trigger beat analysis pipeline, then re-read the beat times from stimulus data.
        self.analyze_beats(stimulus, psth_dt=psth_dt, psth_sigma=psth_sigma, padding=padding)
        _, onsets, offsets = self.find_beat_times(plot=False)

        onsets = np.asarray(onsets, dtype=float)
        offsets = np.asarray(offsets, dtype=float)
        if len(onsets) < 2 or len(offsets) == 0:
            if show:
                plt.show()
            return fig

        avg_ioi = float(np.mean(np.diff(onsets)))
        print(f'Average IOI: {avg_ioi}')
        n_pairs = min(len(offsets), len(onsets) - 1)

        raster_ax = fig.axes[1]
        psth_ax = fig.axes[-1]

        # Shade gap windows by whether they are longer/shorter than the average IOI.
        for i in range(n_pairs):
            gap_start = onsets[i]
            gap_end = onsets[i + 1]
            if gap_end <= gap_start:
                continue

            gap_duration = gap_end - gap_start
            color = 'green' if gap_duration > avg_ioi else 'red'

            raster_ax.axvspan(gap_start, gap_end, color=color, alpha=0.15)
            psth_ax.axvspan(gap_start, gap_end, color=color, alpha=0.15)

        if show:
            plt.show()

        return fig

    def _compare_offsets(self, stim, n_beats=3, output_path=None, plot=True, predict_onset=False):
        import matplotlib.pyplot as plt
        import numpy as np

        # --- Get trial and baseline PSTHs ---
        _, trial_spikes, _ = self.raster(stim, ax=None, baseline=False, plot=False, padding=1)
        _, baseline_spikes, _ = self.raster(stim, ax=None, baseline=True, plot=False, padding=1)

        _, trial_psth = self.psth(stim, trial_spikes, ax=None, normalize=True, dt=0.001, sigma=0.015, plot=False)
        _, baseline_psth = self.psth(stim, baseline_spikes, ax=None, normalize=True, dt=0.001, sigma=0.015, plot=False)

        rs = trial_psth[1] - baseline_psth[1]
        time = trial_psth[0]

        # --- Stimulus trace ---
        stim_ax, self.stimulus_time, self.stimulus_data = self.plot_stimulus(stim, plot=False, ax=None)

        # --- Beat times ---
        _, onsets, offsets = self.find_beat_times(plot=False)
        onsets = np.array(onsets) - self.padding
        offsets = np.array(offsets) - self.padding

        ioi = np.mean(np.diff(offsets[:3]))
        beat_duration = np.mean(offsets[:3] - onsets[:3])

        # --- Detect omissions ---
        offset_diffs = np.diff(offsets)
        mask = offset_diffs > 1.5 * ioi
        omission_anchor_idx = np.where(mask)[0]
        offset_response_locs = np.append(offsets[:-1][mask], offsets[-1])
        omission_counts = {
            idx: max(1, int(np.round(offset_diffs[idx] / ioi)) - 1)
            for idx in omission_anchor_idx
        }

        # No vertical omission markers on the waveform panel.

        # --- Colors for different PSTHs ---
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        stim_line_segments = []
        omission_line_segments = []
        aligned_curves = []
        first_real_onsets = []

        for i, offset_response_loc in enumerate(offset_response_locs):
            # --- Last n beats before omission ---
            beat_idx = np.where(offsets <= offset_response_loc)[0]
            beats = offsets[beat_idx[-n_beats:]]
            beats_onsets = onsets[beat_idx[-n_beats:]]
            beats_offsets = offsets[beat_idx[-n_beats:]]

            # Include omitted beat for alignment
            beats = np.append(beats, offset_response_loc)

            # PSTH window: from first beat onset to 1 s after last beat
            mask_window = (time >= beats_onsets[0]) & (time <= offset_response_loc + 1)
            psth_window = rs[mask_window]
            t_window = time[mask_window]

            # Align to last beat in window
            ref = beats[-1]
            t_rel = t_window - ref

            aligned_curves.append((t_rel.copy(), psth_window.copy()))

            # PSTH color
            color = colors[i % len(colors)]
            ax.plot(t_rel, psth_window, alpha=0.8, color=color, label=f'Omission {i + 1}')
            if i == 1:
                omit_psth = psth_window
                omit_time = t_rel

            # --- Shaded regions: beats before the omission ---
            pre_omission_idx = np.where(offsets <= offset_response_loc)[0]
            for start, end in zip(onsets[pre_omission_idx][-n_beats:], offsets[pre_omission_idx][-n_beats:]):
                x0, x1 = start - ref, end - ref
                ax.axvspan(x0, x1, color='gray', alpha=0.1)
                stim_line_segments.append((x0, x1, 'gray'))

            # --- Shaded regions: beats after the omission ---
            post_omission_idx = np.where(onsets > offset_response_loc)[0]
            if len(post_omission_idx) > 0:
                first_real_onsets.append(onsets[post_omission_idx[0]] - ref)
            for start, end in zip(onsets[post_omission_idx], offsets[post_omission_idx]):
                x0, x1 = start - ref, end - ref
                ax.axvspan(x0, x1, color='gray', alpha=0.1)
                stim_line_segments.append((x0, x1, 'gray'))

            # Expected omitted beat timing(s): support one or more omitted beats.
            this_idx = beat_idx[-1] if len(beat_idx) > 0 else None
            n_omitted = omission_counts.get(this_idx, 0)
            for omission_num in range(1, n_omitted + 1):
                omission_x0 = omission_num * ioi - beat_duration
                omission_x1 = omission_num * ioi
                omission_line_segments.append((omission_x0, omission_x1))
                ax.axvspan(omission_x0, omission_x1, color='red', alpha=0.04)

            # Last beat marker
            # ax.axvline(x=0, color='gray', linestyle='-', linewidth=2)

        if predict_onset:
            pass

        # Draw short guide lines below the PSTH to mark stimulus windows.
        if stim_line_segments:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min if y_max > y_min else 1.0
            stim_line_y = y_min - 0.06 * y_range

            for x0, x1, line_color in stim_line_segments:
                ax.hlines(stim_line_y, x0, x1, color=line_color, linewidth=3, alpha=0.95)

            for x0, x1 in omission_line_segments:
                ax.hlines(stim_line_y, x0, x1, color='red', linewidth=3, alpha=0.95)

            ax.set_ylim(stim_line_y - 0.05 * y_range, y_max)

        # --- Beautify plot ---
        ax.set_xlabel('Time relative to last beat (s)')
        ax.set_ylabel('ΔFiring rate (trial - baseline)')
        ax.set_title(f'{stim}: Aligned PSTH around omission (last {n_beats} beats)')
        ax.grid(True, alpha=0.3)
        # ax.legend()
        plt.tight_layout()
        ax.set_xlim(-0.5, 1)

        if output_path is not None:
            plt.savefig(output_path, format='png')

        if plot:
            plt.show()

        # Compute omission-window difference between the first two aligned PSTH lines.
        # Window is t=0 to first real beat onset (t>0).
        omission_difference = None
        if len(aligned_curves) >= 2 and len(first_real_onsets) > 0:
            window_end = min([x for x in first_real_onsets if x > 0], default=None)
            if window_end is not None:
                t1, y1 = aligned_curves[0]
                t2, y2 = aligned_curves[1]

                start = 0.0
                end = float(window_end)

                # Restrict to the valid overlap and interpolate both lines onto a common grid.
                t_start = max(start, float(np.min(t1)), float(np.min(t2)))
                t_end = min(end, float(np.max(t1)), float(np.max(t2)))

                if t_end > t_start:
                    n_pts = 400
                    t_common = np.linspace(t_start, t_end, n_pts)
                    y1_i = np.interp(t_common, t1, y1)
                    y2_i = np.interp(t_common, t2, y2)
                    diff = y1_i - y2_i

                    omission_difference = {
                        "time": t_common,
                        "diff": diff,
                        "mean_diff": float(np.mean(diff)),
                        "mean_abs_diff": float(np.mean(np.abs(diff))),
                        "integral_abs_diff": float(np.trapezoid(np.abs(diff), t_common)),
                        "window": (t_start, t_end),
                    }

        return omission_difference

    def responsive_significance_test(self, stim):
        trial_spikes, trial_duration = self.raster(stim, baseline=False, plot=False, ax=None)
        baseline_spikes, baseline_duration = self.raster(stim, baseline=True, plot=False, ax=None)
