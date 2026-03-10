import pandas as pd
import os
import h5py
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import threading

def label_neuron(id, session_id, unit_id, spike_file, stimulus_file, stim_lib_path):
    samplerate=30000
    stimuli_arr = pd.read_json(os.path.join(stimulus_file, '..', 'stimuli.json'))
    filtered = stimuli_arr[stimuli_arr['Stimuli Type'].str.contains('CON|White Noise', regex=True)].reset_index(drop=True)

    with h5py.File(spike_file, 'r') as f:
        spiketimes = f[f'unit_{unit_id}'][:]/samplerate

    fig, ax = plt.subplots(3 * len(pd.unique(filtered['Stimuli Type'])) , 1,
                           figsize=(5, len(pd.unique(filtered['Stimuli Type'])) * 3), sharex=True)
    for s, stim in enumerate(pd.unique(filtered['Stimuli Type'])):
        reps = filtered[filtered['Stimuli Type'] == stim].reset_index(drop=True)

        avg_duration = np.mean(reps['Duration'])
        stim_pad = (5 - avg_duration)/2

        stim_path = os.path.join(stim_lib_path, f'{stim.rstrip('.wav')}.wav')
        sampling_rate, data = wav.read(stim_path)
        data = data.astype(float)
        data /= np.max(np.abs(data))
        duration = data.shape[0] / sampling_rate
        t = np.linspace(0, duration, data.shape[0])
        ax[s*3].plot(t+stim_pad, data)


        all_spks = []
        window_duration = 5
        for r, row in reps.iterrows():
            stim_duration = row['Duration']
            pad = (5 - stim_duration)/2
            start_time = row['Start Time'] - pad
            end_time = row['End Time'] + pad
            mask = (spiketimes >= start_time) & (spiketimes < end_time)
            trial_spks = spiketimes[mask] - start_time

            ax[3*s+1].vlines(trial_spks, ymin=r, ymax=r + 0.5)

            all_spks.extend(trial_spks)
        ax[3 * s].set_title(stim)

        bins = np.arange(0, 5 + 10 / 1000, 10 / 1000)
        counts, edges = np.histogram(sorted(all_spks), bins=bins)
        ax[3*s+2].plot(edges[:-1], counts)

    label = {"value": None}

    def on_key(event):
        if event.key in ['a', 'n', 'u', 'q']:
            label['value'] = event.key
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    fig.subplots_adjust(
        top=0.95,  # top margin
        bottom=0.05,  # bottom margin
        left=0.08,  # left margin
        right=0.95,  # right margin
        hspace=0.5,  # vertical space between subplots
        wspace=0.3  # horizontal space (usually irrelevant for 1-column)
    )

    plt.show()  # <-- BLOCKS correctly in Qt
    return label['value']

def count_unlabeled(cursor):
    cursor.execute('''
        SELECT COUNT(*)
        FROM neurons
        WHERE label IS NULL OR label = ''
    ''')
    return cursor.fetchone()[0]

def curate_responses(db_path, stim_lib_path, isi_cutoff=1):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    total_unlabeled = count_unlabeled(cursor)
    print(f"Starting curation. Unlabeled neurons: {total_unlabeled}")

    # ---- TWO PASSES: good ISI first, then bad ----
    passes = [
        ("Good ISI (< cutoff)", "manual_isi_1_5 < ?", (isi_cutoff,)),
        ("Poor ISI (>= cutoff)", "manual_isi_1_5 >= ?", (isi_cutoff,))
    ]

    for pass_name, isi_clause, params in passes:
        cursor.execute(f'''
            SELECT id, session_id, unit_id, spike_file, stimulus_file
            FROM neurons
            WHERE (label IS NULL OR label = 'u')
              AND {isi_clause}
            ORDER BY manual_isi_0_7 ASC
        ''', params)

        rows = cursor.fetchall()
        print(f"\n{pass_name}: {len(rows)} neurons")

        for idx, (id, session_id, unit_id, spike_file, stimulus_file) in enumerate(rows, 1):
            print(f"[{pass_name}] {idx}/{len(rows)} — remaining unlabeled: {len(rows) - idx}")

            key = label_neuron(id, session_id, unit_id, spike_file, stimulus_file, stim_lib_path)

            if key == 'q':
                print("Stopping curation early.")
                conn.close()
                return

            label_map = {
                'a': 'auditory',
                'n': 'non_auditory',
                'u': 'u'
            }

            if key not in label_map:
                continue

            cursor.execute(
                '''
                UPDATE neurons
                SET label = ?
                WHERE id = ?
                  AND session_id = ?
                  AND unit_id = ?
                ''',
                (label_map[key], id, session_id, unit_id)
            )
            conn.commit()  # safe resume

    conn.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--database_path", required=True)
    parser.add_argument("--isi_cutoff", type=int, default=1)
    parser.add_argument("--stimulus_library_path", required=True)
    args = parser.parse_args()

    curate_responses(args.database_path, args.stimulus_library_path, isi_cutoff=args.isi_cutoff)