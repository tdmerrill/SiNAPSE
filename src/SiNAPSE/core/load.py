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
        import subprocess

        cmd = [
            'python',
            'GUI/LabelAuditoryNeurons.py',
        ]
        p = subprocess.Popen(cmd)

