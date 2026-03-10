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

    def __init__(self, recording_fp, samplerate, db:Database):
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
    def find_probe(self):
        """
        Find probe from recording name.

        Returns
        ----------
        key: str
            cambridge neurotech probe name (e.g. ASSY-37-H4)
        """

        import json, os

        # Path relative to this file
        json_path = os.path.join(os.path.dirname(__file__), '..', 'experiment', 'probes.json')

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

        self.probe_id = probe_id
        json_path = os.path.join(os.path.dirname(__file__), '..', 'experiment', 'probes.json')
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

        for line in p.stdout:
            print("[worker]", line, end="")

        p.wait()

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
                raise RuntimeError('Probe cannot be detected. Call Recording.set_probe_id(probe_id) to set the probe manually.')
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

        sort_worker_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'workers', 'sort_data.py')
        )
        print(sort_worker_path)
        cmd = [
            sys.executable,
            sort_worker_path,
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

        threading.Thread(target=self._wait_for_process, args=(p,), daemon=True).start()

        self.db.calculate_isi_violations()