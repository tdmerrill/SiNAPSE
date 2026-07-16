class Database:
    """
        A class for handling SQL Database operations.

        Attributes
        ----------
        db_path : str
            Path to the database.db file.
        stim_lib : str
            Path to stimulus library with .wav file for all stimuli.
    """

    def __init__(self, db_path, stim_lib):
        """
            Initialize a Database object.

            Parameters
            ----------
            db_path : str
                Path to the database.db file.
            stim_lib : str
                Path to stimulus library with .wav file for all stimuli.
        """

        self.db_path = db_path
        self.stim_library = stim_lib

        self.initialize_database()

        self.X = None

    def load_neurons_from_database(self, select_columns, conditions):
        """
           load data from sql database that fits given conditions

           Parameters
           ----------
           select_columns : list
               List of neuron features to return
           conditions : {'feature', 'condition'}
               conditions for neurons that will be returned (ex. manual_isi_1 < 1)

           Returns
           -------
           neurons : np.ndarray
               neurons that fit the criteria
           """

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        select_sql = ", ".join(select_columns)

        query = f"SELECT {select_sql} FROM neurons"
        values = []

        if conditions:
            where_clauses = []

            for col, (op, val) in conditions.items():
                where_clauses.append(f"{col} {op} ?")
                values.append(val)

            where_sql = " AND ".join(where_clauses)
            query += f" WHERE {where_sql}"

        cursor.execute(query, values)

        rows = cursor.fetchall()

        conn.close()

        return rows

    def initialize_database(self):
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # One row per neuron (logical unit)
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS neurons
                    (
                        id
                        INTEGER
                        PRIMARY
                        KEY
                        AUTOINCREMENT,

                        --Identity
                        session_id
                        TEXT,
                        probe
                        TEXT,
                        unit_id
                        INTEGER,

                        --Unit information
                        snr
                        REAL,
                        firing_rate
                        REAL,
                        isi_violation_ratio
                        REAL,
                        presence_ratio
                        REAL,
                        sliding_rp_violation
                        REAL,
                        drift
                        REAL,
                        amplitude_median
                        REAL,
                        amplitude_cv
                        REAL,
                        noise_cutoff
                        REAL,
                        spike_width_pp
                        REAL,
                        spike_width_hw
                        REAL,
                        unit_loc_x
                        REAL,
                        unit_loc_y
                        REAL,
                        manual_isi_0_7
                        REAL,
                        manual_isi_1
                        REAl,
                        manual_isi_1_5
                        REAL,
                        label
                        TEXT,
                        stimulus_responsive
                        TEXT,


                        --Paths to external data
                        spike_file
                        TEXT,
                        stimulus_file
                        TEXT,

                        UNIQUE
                    (
                        session_id,
                        probe,
                        unit_id
                    )
                        )
                    """)

        conn.commit()
        conn.close()

    def calculate_isi_violations(self):
        import sqlite3
        import numpy as np
        import h5py

        print("adding manual isi where it's missing!")
        db_path = self.db_path
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        cur.execute('''
                    SELECT id, session_id, unit_id, spike_file
                    FROM neurons
                    WHERE (
                        manual_isi_0_7 IS NULL
                            OR manual_isi_1 IS NULL
                            OR manual_isi_1_5 IS NULL
                        )
                      AND spike_file IS NOT NULL
                    ''')
        neurons = cur.fetchall()

        if len(neurons) == 0:
            print("No neurons found")
        else:
            for n, neuron in enumerate(neurons):
                id = neuron[0]
                session_id = neuron[1]
                unit_id = neuron[2]
                spike_file = neuron[3]

                print(f'working on neuron {n + 1}/{len(neurons)}')

                data_dict = {}
                with h5py.File(spike_file, "r") as f:
                    for key in f.keys():
                        data_dict[key] = f[key][:]

                spikes = np.array(data_dict[f'unit_{unit_id}']) / 30000
                isi = np.array(np.diff(spikes))
                violation_rate_1_5 = len(isi[isi < 1.5 / 1000]) / len(spikes) * 100
                violation_rate_1 = len(isi[isi < 1 / 1000]) / len(spikes) * 100
                violation_rate_0_7 = len(isi[isi < 0.7 / 1000]) / len(spikes) * 100

                cur.execute(
                    '''
                    UPDATE neurons
                    SET manual_isi_1_5 = ?,
                        manual_isi_1   = ?,
                        manual_isi_0_7 = ?
                    WHERE id = ?
                      AND session_id = ?
                      AND unit_id = ?
                    ''',
                    (violation_rate_1_5, violation_rate_1, violation_rate_0_7, id, session_id, unit_id)
                )
        con.commit()
        con.close()

    def label_auditory_neurons(self):
        import subprocess, os

        label_worker_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'workers', 'auditory_labeling.py')
        )
        cmd = [
            'python',
            label_worker_path,
            '--database_path', self.db_path,
            '--stimulus_library_path', self.stim_library,
        ]
        p = subprocess.Popen(cmd)

        p.wait()

    def label_neuron_locations(self):
        import os, json

        neuron_dir_path = os.path.join(self.db_path, '..')
        self.neuron_loc_path = os.path.join(self.db_path, '..', 'neuron_locations.json')
        if not os.path.exists(neuron_dir_path):
            os.makedirs(neuron_dir_path)

        conditions = {
            'manual_isi_1': ('<', 1),
            'label': ('=', 'auditory'),
        }
        self.update_location_labeling(conditions=conditions)

        with open(self.neuron_loc_path, 'r') as file:
            data = json.load(file)
        unlabeled_recs = []
        for recording in data.keys():
            for neuron in data[recording].keys():
                if data[recording][neuron] == "" and recording not in unlabeled_recs:
                    unlabeled_recs.append(recording)
        chosen_rec = self.choose_recording(unlabeled_recs)
        neurons = data[chosen_rec]
        labels = self.label_locs_gui(neurons)
        data[chosen_rec] = labels
        json.dump(data, open(self.neuron_loc_path, 'w'))

    def label_stimulus_neurons(self):
        import subprocess, os

        label_worker_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'workers', 'stimulus_responsive_labeling.py')
        )
        cmd = [
            'python',
            label_worker_path,
            '--database_path', self.db_path,
            '--stimulus_library_path', self.stim_library,
        ]
        p = subprocess.Popen(cmd)

        p.wait()

    def update_location_labeling(self, recording_col='session_id', neuron_col='unit_id', select_columns=None,
                                 conditions=None):
        """
            Create/update a JSON file mapping:
            recording -> neuron_id -> ""

            Only adds new recordings/neurons, does NOT overwrite existing labels.
            """

        import json
        import os

        json_path = self.neuron_loc_path
        # -----------------------
        # 1. Load existing JSON
        # -----------------------
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # -----------------------
        # 2. Query database
        # -----------------------
        if select_columns is None:
            select_columns = [recording_col, neuron_col]

        rows = self.load_neurons_from_database(
            select_columns=select_columns,
            conditions=conditions
        )

        # figure out column indices
        rec_idx = select_columns.index(recording_col)
        neuron_idx = select_columns.index(neuron_col)

        # -----------------------
        # 3. Populate structure
        # -----------------------
        for row in rows:
            recording = str(row[rec_idx])
            neuron = str(row[neuron_idx])

            # add recording if missing
            if recording not in data:
                data[recording] = {}

            # add neuron if missing
            if neuron not in data[recording]:
                data[recording][neuron] = ""  # empty label

        # -----------------------
        # 4. Save JSON
        # -----------------------
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Updated JSON saved to: {json_path}")

    @staticmethod
    def choose_recording(options, n_cols=4):
        import tkinter as tk
        import math

        if not options:
            print("No recordings available")
            return None

        selected = {"value": None}

        def set_choice(choice):
            print("Selected:", choice)
            selected["value"] = choice
            root.destroy()

        root = tk.Tk()
        root.title("Select Recording")

        # --- create grid of buttons ---
        for i, opt in enumerate(options):
            row = i // n_cols
            col = i % n_cols

            btn = tk.Button(
                root,
                text=opt,
                width=50,
                command=lambda o=opt: set_choice(o)
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        # --- make grid expand nicely ---
        n_rows = math.ceil(len(options) / n_cols)

        for c in range(n_cols):
            root.grid_columnconfigure(c, weight=1)

        for r in range(n_rows):
            root.grid_rowconfigure(r, weight=1)

        root.mainloop()
        return selected["value"]

    @staticmethod
    def label_locs_gui(neurons):
        import tkinter as tk

        labels = ["", "NCM", "Field L", "ORR", "HVC", "Area X"]

        # neurons is now a dict: {neuron_id: label}
        data = dict(neurons)  # copy so we don’t modify input directly

        # color map
        color_map = {
            "": "lightgray",
            "NCM": "lightblue",
            "Field L": "lightgreen",
            "ORR": "lightyellow",
            "HVC": "lightpink",
            "Area X": "lightcoral"
        }

        def cycle_label(n):
            current = data[n]
            idx = labels.index(current)
            new_label = labels[(idx + 1) % len(labels)]
            data[n] = new_label

            buttons[n]["text"] = f"{n}: {new_label}"
            buttons[n]["bg"] = color_map.get(new_label, "white")

        root = tk.Tk()
        root.title("Label Neurons")

        buttons = {}

        for i, n in enumerate(neurons):
            label = data[n]

            btn = tk.Button(
                root,
                text=f"{n}: {label}",
                width=14,
                bg=color_map.get(label, "white"),
                command=lambda x=n: cycle_label(x)
            )

            btn.grid(row=i // 5, column=i % 5, padx=5, pady=5)
            buttons[n] = btn

        root.mainloop()
        return data

    @property
    def recordings(self, select_columns=['session_id']):
        # First condition: label == 'auditory'
        cond1 = {
            'manual_isi_1': ('<', 1),
            'label': ('=', 'auditory'),
        }

        # Second condition: stim_responsive == 1
        cond2 = {
            'manual_isi_1': ('<', 1),
            'stim_responsive': ('=', 1),
        }

        neurons1 = self.load_neurons_from_database(select_columns, cond1)
        neurons2 = self.load_neurons_from_database(select_columns, cond2)

        # Combine and deduplicate
        neurons = neurons1 + neurons2
        unique = list(set(x[0] for x in neurons))

        return unique

    def detect_stimulus_responses(self, recordings_path, redetect=False, regenerate=False):
        from .spikes import Neuron
        from .sort import Recording

        import os
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        import sqlite3

        db_path = self.db_path
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        cur.execute('''
                    SELECT id, session_id, unit_id
                    FROM neurons
                    WHERE stim_responsive IS NULL
                    ''')
        neurons = cur.fetchall()

        try:
            for neuron in neurons:
                try:
                    id, session_id, unit_number = neuron
                    print(id, session_id, unit_number)
                    bird_id = session_id.split(' ')[0]
                    rec_fp = os.path.join(recordings_path, bird_id, session_id)
                    rec = Recording(rec_fp, samplerate=30 * 1000, db=self)
                    recording_name = session_id
                    N = Neuron(recording_name, unit_number, db=self, rec=rec)
                    stim = N.load

                    responsive = N.is_responsive

                    cur.execute(
                        '''
                        UPDATE neurons
                        SET stim_responsive = ?
                        WHERE (id = ?)
                          AND session_id = ?
                          AND unit_id = ?
                        ''',
                        (responsive, id, session_id, unit_number)
                    )
                    print(f'Setting {session_id}: {unit_number} to responsive={responsive}')
                except OSError as e:
                    print(f'error in {neuron}: {e}')
                except TypeError as e:
                    print(f'error in {neuron}: {e}')
        finally:
            print('closing database')
            con.commit()
            con.close()

    def population_analysis(self, stimulus, recordings_path=None, regenerate=False):
        from .spikes import Neuron
        from .sort import Recording

        import os
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        import sqlite3

        if self.X is None or regenerate:
            if recordings_path is None:
                raise ValueError('recordings_path must be specified')

            db_path = self.db_path
            con = sqlite3.connect(db_path)
            cur = con.cursor()

            cur.execute('''
                        SELECT id, session_id, unit_id
                        FROM neurons
                        WHERE stim_responsive = 1
                        ''')
            neurons = cur.fetchall()

            DURATION = None
            K = 30
            dt = 0.001

            X_list = []  # collect valid neurons

            for neuron in neurons:
                try:
                    id, session_id, unit_number = neuron

                    if ('field l' in session_id.lower() or
                            'fieldl' in session_id.lower() or
                            'field_l' in session_id.lower()):

                        bird_id = session_id.split(' ')[0]
                        rec_fp = os.path.join(recordings_path, bird_id, session_id)

                        rec = Recording(rec_fp, samplerate=30_000, db=self)
                        neuron_obj = Neuron(session_id, unit_number, db=self, rec=rec)

                        stim = neuron_obj.load

                        if stimulus in stim:
                            _, spikes, duration = neuron_obj.raster(
                                stimulus,
                                baseline=False,
                                plot=False,
                                ax=None,
                                separate_trials=True
                            )

                            # Initialize time axis once
                            if DURATION is None:
                                DURATION = duration
                                padding = neuron_obj.padding

                                t = np.arange(-padding, duration + padding + dt, dt)
                                T = len(t) - 1  # histogram output length

                                print(f"Setting global duration: {DURATION}")

                            if len(spikes) == K:
                                neuron_data = np.zeros((K, T))

                                for k, s in enumerate(spikes):
                                    counts, _ = np.histogram(s, bins=t)
                                    neuron_data[k, :] = counts

                                X_list.append(neuron_data)
                except TypeError:
                    continue
            # Stack into final array
            X = np.stack(X_list, axis=0)

            print(f"Final shape: {X.shape}")

            X = X / dt
            self.X = X

        X_avg = X.mean(axis=1)

        from scipy.stats import zscore
        X_norm = zscore(X_avg, axis=1)  # normalize each neuron
        X_pca_input = X_norm.T

        from sklearn.decomposition import PCA

        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_pca_input)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.plot(X_pca[:, 0], X_pca[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Population trajectory')
        plt.show()

        num = 2021
        plt.figure(figsize=(6, 6))

        # full trajectory
        plt.plot(X_pca[:, 0], X_pca[:, 1], alpha=0.3)

        # before omission
        plt.plot(X_pca[:num, 0], X_pca[:num, 1], label='pre-omit')

        # after omission
        plt.plot(X_pca[num:, 0], X_pca[num:, 1], label='post-omit')

        # omission point
        plt.scatter(X_pca[num, 0], X_pca[num, 1], s=80, label='omission')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.title('Population trajectory (omit split)')
        plt.show()

        window = 200  # 200 ms if dt=1 ms

        start = max(0, num - window)
        end = min(len(X_pca), num + window)

        plt.figure(figsize=(6, 6))

        plt.plot(X_pca[start:end, 0], X_pca[start:end, 1])

        # highlight omission
        plt.scatter(X_pca[num, 0], X_pca[num, 1], s=100)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Zoomed trajectory around omission')
        plt.show()

        t_centered = t[:-1] - t[num]

        plt.figure(figsize=(6, 6))

        sc = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=t_centered,
            s=10
        )

        plt.colorbar(sc, label='Time from omission (s)')
        plt.title('Trajectory centered on omission')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

        omit_state = X_pca[num]

        dist = np.linalg.norm(X_pca - omit_state, axis=1)

        plt.plot(t[:-1], dist)
        plt.axvline(t[num], linestyle='--')
        plt.title('Distance from omission state')
        plt.xlabel('Time')
        plt.ylabel('Distance in PCA space')
        plt.show()

    @property
    def metrics(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM neurons")

        # Extract column names from the description
        column_names = [description[0] for description in cursor.description]
        return column_names