from .load import Database


class Recording:
    """
    A class for handling OpenEphys recording files.

    Attributes
    ----------
    recording_fp : str
        Path to the recording folder.
    samplerate : float
        Sampling rate of the recording (usually 30 kHz).
    """

    def __init__(self, recording_fp, samplerate, db: Database):
        """
        Initialize a Recording object.

        Parameters
        ----------
        recording_fp : str
            Path to the recording folder.
        samplerate : float
            Sampling rate of the recording.
        """

        self.rec_fp = recording_fp
        self.log_fp = None
        self.probe_id = None
        self.samplerate = samplerate

        self.db = db

        from pathlib import Path
        self.recording_name = Path(recording_fp).name

    def find_log_file(self, recording_name):
        """
        Search subdirectories for a log file path with the recording name.

        Parameters
        ----------
        recording_name : str
            Name of the recording folder/log file to search for.

        Returns
        -------
        file: str
            Path to the log file path.
        """

        from pathlib import Path

        log_name = recording_name.split('(')[0].rstrip(' ')
        print(log_name)

        directory = Path(self.rec_fp)

        for file in directory.rglob("*"):
            if file.is_file() and file.stem == log_name:
                return file

        return None

    @property
    def load_log_file(self):
        """
        Load log file contents.

        Returns
        -------
        df: DataFrame
            columns: Start Time, End Time, Stimulus, Delay Post.
        """

        import pandas as pd
        from pathlib import Path

        rec_name = Path(self.rec_fp).name
        self.log_fp = self.find_log_file(rec_name)

        if self.log_fp is not None:
            df = pd.read_csv(self.log_fp, header=1)
        else:
            raise FileNotFoundError(f"Cannot find log file for {rec_name}")

        return df

    @property
    def unfiltered_log_stim_order(self):
        import pandas as pd
        from pathlib import Path
        import os

        unfiltered_path = os.path.join(self.rec_fp, 'filtered')
        logs = [
            f for f in os.listdir(unfiltered_path)
            if os.path.isfile(os.path.join(unfiltered_path, f))
        ]
        df = pd.read_csv(os.path.join(unfiltered_path, logs[0]), header=1)
        return df['Stimulus']

    @property
    def find_probe(self):
        """
        Find probe from recording name.

        Returns
        ----------
        key: str
            cambridge neurotech probe name (e.g. ASSY-37-H4)
        """

        import json, os
        from importlib.resources import files

        # Path relative to this file
        json_path = files("SiNAPSE.experiment").joinpath("probes.json")
        # Load JSON
        with open(json_path, 'r') as f:
            probes_dict = json.load(f)

        # Loop through keys and check the 'name' field
        for key, info in probes_dict.items():
            if info['name'] in self.recording_name:
                print(f"Found probe key: {key}")
                self.probe_id = key
                self.channel_map = probes_dict[key]['channel_map']
                return key  # Return the original JSON key

        # If no match
        return None

    def set_probe_id(self, probe_id):
        """
        Allows user to manually set probe id if it is not found in the recording name.

        Parameters
        ----------
        probe_id : str
            Probe ID.
        """

        import os, json
        from importlib.resources import files

        self.probe_id = probe_id
        json_path = files("SiNAPSE.experiment").joinpath("probes.json")
        with open(json_path, 'r') as f:
            probes_dict = json.load(f)
        self.channel_map = probes_dict[probe_id]['channel_map']

        print(f'probe id set to {self.probe_id}')
        print(f'channel map set to {self.channel_map}')

    @staticmethod
    def _wait_for_process(p):
        """
        Waits until the subprocess is finished, then continues.

        Parameters
        ----------
        p : subprocess.Popen
            subprocess.Popen object.
        """

    def sort(self, local_path=None):
        """
        Sorts a OpenEphys recording file.

        Parameters
        ----------
        local_path : str
            Path to the local directory for copying neural data.
        """

        import os, json, subprocess, threading, sys

        if self.probe_id is None:
            probe_id = self.find_probe
            if probe_id is None:
                raise RuntimeError(
                    'Probe cannot be detected. Call Recording.set_probe_id(probe_id) to set the probe manually.')
            else:
                print(f'Automatically detected probe id: {probe_id}')
        else:
            print(f'Detected probe id: {self.probe_id}')
            probe_id = self.probe_id

        if local_path is None:
            home_dir = os.path.expanduser('~')
            base_folder = os.path.join(home_dir, '.SiNAPSE', 'local')
            os.makedirs(base_folder, exist_ok=True)
            local_path = base_folder
            print(f'Automatically saving files to {local_path}')

        # local_path = os.path.join(local_path, self.recording_name, 'unfiltered')

        sorting_path = os.path.join(self.rec_fp, 'unfiltered')
        channel_map = self.channel_map

        # sort_worker_path = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), '..', 'workers', 'sort_data.py')
        # )
        # print(sort_worker_path)

        import SiNAPSE.workers.sort_data as mod
        import os
        script_path = mod.__file__
        print("SCRIPT PATH:", script_path)
        print("EXISTS:", os.path.exists(script_path))
        cmd = [
            sys.executable,
            script_path,
            '--data', sorting_path,
            '--probe', probe_id,
            '--chanmap', json.dumps(channel_map),
            "--sorter", "kilosort4",
            "--destination", local_path,
            '--database_path', self.db.db_path,
        ]
        # print("COMMAND:", cmd)
        # print('running subprocess')
        import subprocess, sys

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        for line in p.stdout:
            print("[sorter]", line, end="")

        p.wait()
        #
        # threading.Thread(target=self._wait_for_process, args=(p,), daemon=True).start()

        self.db.calculate_isi_violations()

    def plot_all_neurons(self, stim, output_path=None, baseline=True, padding=0.5, format='png'):
        if not output_path is None:
            from .spikes import Neuron
            from pathlib import Path

            select_cols = ['unit_id', 'session_id']
            conditions = {
                'manual_isi_0_7': ('<', 1),
                'stim_responsive': ('=', 1),
                'session_id': ('=', f'{Path(self.rec_fp).name}'),
            }
            neurons = self.db.load_neurons_from_database(select_cols, conditions)
            for neuron in neurons:
                unit_number, recording_name = neuron
                N = Neuron(recording_name, unit_number, db=self.db, rec=self)
                stims = N.load
                if stim in stims:
                    N.plot(stimuli_to_raster=[stim], baseline=baseline, padding=padding)
                    N.save_plots(output_path=output_path, format=format, name_prefix=f'unit{unit_number}_')
                else:
                    print(f'Missing stimulus: {stim}')

        else:
            raise FileNotFoundError('Please specify an output path using the `output_path` argument.')

    @property
    def session_path(self):
        import os
        recording_path = self.rec_fp
        unfiltered_path = os.path.join(recording_path, 'filtered')
        dirs = [
            d for d in os.listdir(unfiltered_path)
            if os.path.isdir(os.path.join(unfiltered_path, d))
        ]

        sess_path = os.path.join(unfiltered_path,
                                 dirs[0])  # recordnode_path = os.path.join(sess_path, os.listdir(sess_path)[0])
        # experiment_path = os.path.join(recordnode_path, os.listdir(recordnode_path)[0])
        # rec_path = os.path.join(experiment_path, os.listdir(experiment_path)[0])
        # continuous_path = os.path.join(rec_path, 'continuous')
        # intan_path = os.path.join(continuous_path, os.listdir(continuous_path)[0])
        # dat_path = os.path.join(continuous_path, 'continous.dat')
        return sess_path

    @property
    def start_time(self):
        import os
        sess_path = self.session_path
        recordnode_path = os.path.join(sess_path, os.listdir(sess_path)[0])
        experiment_path = os.path.join(recordnode_path, os.listdir(recordnode_path)[0])
        rec_path = os.path.join(experiment_path, os.listdir(experiment_path)[0])
        sync_messages_path = os.path.join(rec_path, 'sync_messages.txt')
        recording_start_time = None
        try:
            with open(sync_messages_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    recording_start_time = int(parts[-1])  # last line wins
        except FileNotFoundError:
            raise FileNotFoundError(f"sync_messages.txt not found at {sync_messages_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading start time: {e}")
        return recording_start_time

    @property
    def ttl_times(self):
        from open_ephys.analysis import Session
        import numpy as np

        session_path = self.session_path
        session = Session(session_path)
        recording = session.recordnodes[0].recordings[0]
        sample_time = ((recording.events.sample_number) - int(self.start_time)) / 30000

        diffs = np.diff(sample_time)
        mask = (diffs > 0.15) & (diffs < 4.9)  # filter between 0.15 and 5 seconds
        start_time = sample_time[:-1][mask].reset_index(drop=True)
        end_time = sample_time[1:][mask].reset_index(drop=True)
        durations = diffs[mask]

        return start_time, end_time, durations

    @staticmethod
    def get_channel_plot_order(json_path, n_channels, block="0"):
        import json
        import numpy as np

        with open(json_path, "r") as f:
            cfg = json.load(f)

        mapping = np.array(cfg[block]["mapping"])

        # IMPORTANT:
        # Only use mappings that exist in recorded data
        valid = mapping < n_channels
        mapping_valid = mapping[valid]

        # If nothing matches, fall back safely
        if len(mapping_valid) == 0:
            return np.arange(n_channels)

        # Convert to plotting order (physical layout sorted)
        plot_order = np.argsort(mapping_valid)

        return plot_order

    def plot_MUA_power(self, stim, plot_type='heatmap', chan_map_path=None):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from open_ephys.analysis import Session
        from scipy.ndimage import gaussian_filter1d

        # -----------------------
        # Load data
        # -----------------------
        session = Session(self.session_path)
        recording = session.recordnodes[0].recordings[0]
        continuous = recording.continuous[0]

        signal = continuous.samples  # (time × channels)

        try:
            fs = continuous.metadata['sample_rate']
        except:
            fs = continuous.metadata.sample_rate

        timestamps = continuous.timestamps

        def time_to_idx(t):
            return np.searchsorted(timestamps, t)

        n_channels = signal.shape[1]

        # -----------------------
        # Stim table
        # -----------------------
        start_time, end_time, durations = self.ttl_times
        stim_order = self.unfiltered_log_stim_order

        stimuli_arr = pd.DataFrame({
            "Start Time": start_time,
            "End Time": end_time,
            "Duration": durations,
            "Stimuli Type": np.array(stim_order)
        })

        trials = stimuli_arr[stimuli_arr["Stimuli Type"] == stim].reset_index(drop=True)

        padding = 0.5

        # -----------------------
        # COMPUTATION (UNCHANGED)
        # -----------------------
        channel_means = []

        for ch in range(n_channels):
            print(f'working on ch {ch}')
            trial_traces = []

            for _, row in trials.iterrows():
                t0 = row["Start Time"] - padding
                t1 = row["End Time"] + padding

                i0 = time_to_idx(t0)
                i1 = time_to_idx(t1)

                segment = signal[i0:i1, ch]

                mua = np.abs(segment)
                mua = gaussian_filter1d(mua, sigma=fs * 0.002)

                trial_traces.append(mua)

            min_len = min(len(tr) for tr in trial_traces)
            aligned = np.array([tr[:min_len] for tr in trial_traces])

            channel_means.append(aligned.mean(axis=0))

        channel_means = np.array(channel_means)  # (channels × time)

        # -----------------------
        # TIME AXIS (UNCHANGED)
        # -----------------------
        t0 = trials.iloc[0]["Start Time"] - padding
        mask = (timestamps >= t0)
        time_axis = timestamps[mask][:channel_means.shape[1]] - t0

        # -----------------------
        # 🔥 ONLY CHANGE: PLOTTING ORDER
        # -----------------------
        if chan_map_path is not None:
            plot_order = self.get_channel_plot_order(
                chan_map_path,
                n_channels
            )
        else:
            plot_order = np.arange(n_channels)

        plot_data = channel_means[plot_order]

        # -----------------------
        # PLOTTING
        # -----------------------
        if plot_type == "heatmap":
            plt.figure(figsize=(8, 6))

            plt.imshow(
                plot_data,
                aspect="auto",
                origin="lower",
                extent=[
                    time_axis[0],
                    time_axis[-1],
                    0,
                    plot_data.shape[0],
                ],
            )

            plt.colorbar(label="MUA Power")
            plt.xlabel("Time (s)")
            plt.ylabel("Channel (mapped order)")
            plt.title(f"MUA Heatmap (avg across trials) - {stim}")
            plt.show()

        elif plot_type == "stacked":
            plt.figure(figsize=(8, 6))

            offset = np.max(plot_data) * 2

            for ch in range(plot_data.shape[0]):
                plt.plot(
                    time_axis,
                    plot_data[ch] + ch * offset,
                    linewidth=1,
                )

            plt.xlabel("Time (s)")
            plt.ylabel("Channel (mapped order)")
            plt.title(f"Stacked MUA (avg across trials) - {stim}")
            plt.show()

    def probe_pcc(self, stim):
        from .spikes import Neuron
        from pathlib import Path

        select_cols = ['unit_id', 'session_id']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'stim_responsive': ('=', 1),
            'session_id': ('=', f'{Path(self.rec_fp).name}'),
        }
        neurons = self.db.load_neurons_from_database(select_cols, conditions)
        pccs = {}
        for neuron in neurons:
            unit_number, recording_name = neuron
            N = Neuron(recording_name, unit_number, db=self.db, rec=self)
            stims = N.load

            ch = N.peak_channel
            pcc = N.compute_pcc(stim)
            depth = N.get_neuron_data['unit_loc_y']

            if ch not in pccs.keys():
                pccs[ch] = []
            pccs[ch].append((pcc, depth))

        return pccs