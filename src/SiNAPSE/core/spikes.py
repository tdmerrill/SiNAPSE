from .load import Database
from .sort import Recording
import os

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

        self.figure_size = (8,2)
        self.figures = {}

        self.manual_area_curation_path = os.path.join(self.db.db_path, '..', 'neuron_labels.json')

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
    def load_spiketimes(spikes_path, unit_id):
        import h5py

        data_dict = {}
        with h5py.File(spikes_path, "r") as f:
            return f[f'unit_{unit_id}'][:]

    @staticmethod
    def load_stimuli(stim_path):
        import pandas as pd

        return pd.read_json(stim_path)

    def plot(self, raster=True, psth=True, baseline=False, stimuli_to_raster=None, padding = 0.5, normalize_psth=True, psth_sigma=0.015, psth_dt=0.001):
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

        for s, stimulus in enumerate(stimuli_to_raster):
            fig = plt.figure(figsize=(8, 2 * num_rows))
            gs = gridspec.GridSpec(num_rows, 1, figure=fig, hspace=0.5, wspace=0.3)

            axes=[]
            row_idx = 1
            ax_wave = fig.add_subplot(gs[0, 0])
            axes.append(ax_wave)
            self.plot_stimulus(stimulus, ax=ax_wave)
            if raster:
                ax_raster = fig.add_subplot(gs[row_idx, 0])
                axes.append(ax_raster)
                row_idx += 1
            if baseline and raster:
                ax_baseline = fig.add_subplot(gs[row_idx, 0])
                axes.append(ax_baseline)
                row_idx += 1
            if psth:
                ax_psth = fig.add_subplot(gs[row_idx, 0])
                axes.append(ax_psth)
                row_idx += 1

            trial_spikes = None
            baseline_spikes = None

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
                    color='red'
                )

            if psth and trial_spikes is not None:
                _, trial_psth = self.psth(
                    stimulus,
                    trial_spikes,
                    ax=ax_psth,
                    normalize=normalize_psth,
                    dt=psth_dt,
                    sigma=psth_sigma,
                    color='blue'
                )

            if psth and baseline and baseline_spikes is not None:
                _, baseline_psth = self.psth(
                    stimulus,
                    baseline_spikes,
                    ax=ax_psth,
                    normalize=normalize_psth,
                    dt=psth_dt,
                    sigma=psth_sigma,
                    color='red'
                )
            axes = self.format_axes(axes, stimulus, baseline=(baseline and raster))
            fig.suptitle(f"Response of s.u. {self.unit_id} to {stimulus.strip(".wav")}")
            self.figures[stimulus] = fig
            plt.show()

    def plot_stimulus(self, stimulus, ax = None):
        import matplotlib.pyplot as plt
        import scipy.io.wavfile as wav
        import os
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        stim_path = os.path.join(self.db.stim_library, f'{stimulus}.wav')
        sampling_rate, data = wav.read(stim_path)
        data = data.astype(float)
        data /= max(np.abs(data))

        pad_samples = int(self.padding * sampling_rate)
        padded = np.concatenate([
            np.zeros(pad_samples),
            data,
            np.zeros(pad_samples)
        ])

        t = np.arange(len(padded)) / sampling_rate
        ax.plot(t-self.padding, padded, color='black', alpha=0.5)
        ax.set_ylabel('Stimulus Amplitude (a.u.)')

        return ax

    def raster(self, stimulus, baseline = False, plot=True, ax = None, color='black'):

        if plot and ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        stim_times = self.stimuli[self.stimuli['Stimuli Type'] == stimulus].reset_index(drop=True)
        avg_duration = stim_times['Duration'].mean()
        all_trial_spikes = []
        for i, iteration in stim_times.iterrows():
            stim_start_time = iteration['Start Time']
            stim_end_time = iteration['End Time']

            if baseline:
                baseline_start = stim_start_time - avg_duration - 1

                mask = (
                        (self.spiketimes > baseline_start - self.padding) &
                        (self.spiketimes <= baseline_start + avg_duration + self.padding)
                )

                trial_spikes = self.spiketimes[mask] - baseline_start

            else:
                mask = (
                        (self.spiketimes > stim_start_time - self.padding) &
                        (self.spiketimes <= stim_end_time + self.padding)
                )

                trial_spikes = self.spiketimes[mask] - stim_start_time
            all_trial_spikes.extend(trial_spikes)

            if plot:
                ax.vlines(trial_spikes, ymin=i, ymax=i+self.padding, color=color)
                ax.set_ylabel("Trial Number")

        return ax, all_trial_spikes, avg_duration

    def psth(self, stimulus, spikes, ax = None, sigma=0.001, dt=0.015, normalize=True, color='black'):
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        stim_times = self.stimuli[self.stimuli['Stimuli Type'] == stimulus].reset_index(drop=True)
        avg_duration = stim_times['Duration'].mean()

        time_axis = np.arange(-self.padding, avg_duration+self.padding, dt)
        spikes = np.array(spikes)
        n_trials = len(stim_times)
        rate = np.zeros_like(time_axis)
        for spike in spikes:
            rate += np.exp(-(time_axis - spike) ** 2 / (2 * sigma ** 2))

        if normalize:
            rate /= (n_trials * sigma * np.sqrt(2 * np.pi))

        ax.plot(time_axis, rate, color=color)
        ax.set_ylabel("Response (Hz)")
        return ax, [time_axis, rate]

    def format_axes(self, axes, stimulus, baseline=False):
        stim_times = self.stimuli[self.stimuli['Stimuli Type'] == stimulus].reset_index(drop=True)
        duration = stim_times['Duration'].mean()

        for i, ax in enumerate(axes):

            # show only left + bottom spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # choose tick locations
            if baseline:
                ticks = [0, duration]
            else:
                ticks = [0, duration]

            ax.set_xticks(ticks)

            # only bottom subplot gets labels
            if i == len(axes) - 1:
                ax.set_xlabel("Time (s)")
                ax.set_xticklabels([f"{t:.1f}" for t in ticks])
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

        return axes

    def save_plots(self, output_path=None, name_prefix='', format='png', clear_cache=True):
        import os

        if output_path is not None:
            for stim in self.figures.keys():
                self.figures[stim].savefig(os.path.join(output_path, f'{name_prefix}{stim}.{format}'), bbox_inches='tight', format=format)
            if clear_cache:
                self.figures.clear()
        else:
            raise FileNotFoundError('No output path specified')

    @property
    def get_neuron_data(self):
        return(self.neuron_data)

    @property
    def get_manual_label(self):
        from pathlib import Path
        import json

        rec_name = Path(self.rec.rec_fp).name
        bird_id = rec_name.split(' ')[0]

        print(rec_name)
        print(bird_id)

        with open(self.manual_area_curation_path) as f:
            data = json.load(f)

        return data[bird_id][rec_name][f'neuron_{self.unit_id}']