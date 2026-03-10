import os
import shutil
import sys
import argparse, time, pprint

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.full as si
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc
from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe
import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt
import sqlite3, h5py
from scipy.signal import resample_poly
from pathlib import Path

def copy_directory_with_progress(src_path, dst_path):
    """
    Copy the entire directory at src_path into dst_path.
    Shows a terminal progress bar during copy.
    """

    if not os.path.isdir(src_path):
        raise ValueError(f"Source path does not exist or is not a directory: {src_path}")

    # Remove existing destination
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

    # Count files first
    total_files = 0
    for root, dirs, files in os.walk(src_path):
        total_files += len(files)

    if total_files == 0:
        shutil.copytree(src_path, dst_path)
        print("No files to copy. Directory copied.")
        return

    copied_files = 0

    # Create root folder
    os.makedirs(dst_path, exist_ok=True)

    print(f"Copying directory:\n  from: {src_path}\n    to: {dst_path}\n")

    # Copy manually for progress reporting
    for root, dirs, files in os.walk(src_path):

        # Create directories
        rel = os.path.relpath(root, src_path)
        dest_root = os.path.join(dst_path, rel)
        os.makedirs(dest_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_root, file)

            shutil.copy2(src_file, dst_file)

            copied_files += 1
            progress = copied_files / total_files
            bar_len = 40
            filled = int(bar_len * progress)
            bar = "#" * filled + "-" * (bar_len - filled)

            # Print progress bar
            sys.stdout.write(
                f"\r[{bar}] {progress * 100:6.2f}%  "
                f"({copied_files}/{total_files} files)"
            )
            sys.stdout.flush()

    print("\nCopy complete!")

def json_equal(file1, file2):
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    return data1 == data2

def get_unit_locs(analyzer):
    if analyzer.get_extension("unit_locations") is None:
        analyzer.compute("unit_locations")
    unit_locations = analyzer.get_extension("unit_locations").get_data()
    locs = pd.DataFrame(unit_locations)
    return locs[0], locs[1]

def get_log_file_and_stimulus_windows(root_dir, samplerate=30000):
    filtered_rec = os.path.join(root_dir, "filtered")
    unfiltered_rec = os.path.join(root_dir, "unfiltered")

    # --- load log file ---
    csv_files = [
        os.path.join(filtered_rec, f)
        for f in os.listdir(filtered_rec)
    ]
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV log file found in filtered/")
    if len(csv_files) > 1:
        print("Warning: multiple CSV files found, using first one")

    log_path = csv_files[0]
    df = pd.read_csv(log_path, header=1)

    # --- get recording start time from sync_messages ---
    step1 = os.path.join(unfiltered_rec, os.listdir(unfiltered_rec)[0])
    step2 = os.path.join(step1, os.listdir(step1)[0])
    step3 = os.path.join(step2, os.listdir(step2)[0])
    step4 = os.path.join(step3, os.listdir(step3)[0])
    sync_messages_path = os.path.join(step4, 'sync_messages.txt')

    print(f'Sync Messages Filepath: {sync_messages_path}')

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

    print(f"Start Time (samples): {recording_start_time}")

    # --- build session and recording ---
    from open_ephys.analysis import Session
    session_path = os.path.join(unfiltered_rec, os.listdir(unfiltered_rec)[0])
    session = Session(session_path)
    recording = session.recordnodes[0].recordings[0]
    sample_time = ((recording.events.sample_number) - int(recording_start_time)) / 30000

    diffs = np.diff(sample_time)
    mask = (diffs > 0.15) & (diffs < 4.9)  # filter between 0.15 and 5 seconds
    start_time = sample_time[:-1][mask].reset_index(drop=True)
    end_time = sample_time[1:][mask].reset_index(drop=True)
    durations = diffs[mask]

    # ---- and put them all into a dataframe ----
    stimuli_arr = pd.DataFrame(columns=['Start Time', 'End Time', 'Duration', 'Stimuli Type'])
    stimuli_arr['Start Time'] = start_time
    stimuli_arr['End Time'] = end_time
    stimuli_arr['Duration'] = durations
    stimuli_arr['Start Time'] = stimuli_arr['Start Time']
    stimuli_arr['End Time'] = stimuli_arr['End Time']
    stimuli_arr['Stimuli Type'] = np.array(df["Stimulus"])

    stimuli_arr.to_json(os.path.join(root_dir, 'stimuli.json'), orient='records', indent=2)

    return unfiltered_rec

def sync_neurons_for_session(db_path, session_id, probe, analyzer):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    current_units = [int(u) for u in analyzer.sorting.get_unit_ids()]

    # Insert new units
    for unit_id in current_units:
        cur.execute("""
            INSERT INTO neurons (session_id, probe, unit_id)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id, probe, unit_id) DO NOTHING
        """, (session_id, probe, unit_id))

    # Remove units that no longer exist for *this* session+probe only
    if current_units:
        placeholders = ",".join("?" for _ in current_units)
        cur.execute(f"""
            DELETE FROM neurons
            WHERE session_id = ?
              AND probe = ?
              AND unit_id NOT IN ({placeholders})
        """, [session_id, probe, *current_units])
    else:
        cur.execute("""
            DELETE FROM neurons
            WHERE session_id = ? AND probe = ?
        """, (session_id, probe))

    conn.commit()
    conn.close()

def write_spike_times(session_dir, analyzer):
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, 'spikes.h5')

    with h5py.File(path, "w") as f:
        for unit_id in analyzer.sorting.get_unit_ids():
            st = analyzer.sorting.get_unit_spike_train(unit_id)
            f.create_dataset(f"unit_{int(unit_id)}", data=np.asarray(st))

    return path

def write_waveforms(session_dir, recording, sorting_TDC):
    if analyzer_TDC.get_extension("waveforms") is None:
        analyzer_TDC.compute("waveforms")

    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, 'waveforms.h5')

    n_frames = recording.get_num_frames()

    fs = recording.get_sampling_frequency()
    pre = int(0.5 * fs / 1000)
    post = int(1 * fs / 1000)

    with h5py.File(path, "w") as f:
        f.attrs["sampling_frequency"] = fs
        f.attrs["pre_samples"] = pre
        f.attrs["post_samples"] = post

        spike_widths_ms_pp, spike_widths_ms_hw = [], []
        units_grp = f.create_group("units")
        print(f'loading waveform data for {len(sorting_TDC.get_unit_ids())} units')
        for unit_id in sorting_TDC.get_unit_ids():
            spike_times = sorting_TDC.get_unit_spike_train(unit_id)
            unit_wfs = []

            for st in spike_times:
                start = int(st - pre)
                end = int(st + post)
                if start < 0 or end > n_frames:
                    continue
                wf = recording.get_traces(start_frame=start, end_frame=end)
                if wf.shape[0] != (pre + post):
                    continue
                unit_wfs.append(wf)

            if len(unit_wfs) == 0:
                continue

            unit_wfs = np.stack(unit_wfs)
            mean_wf = unit_wfs.mean(axis=0)
            sd_wf = unit_wfs.std(axis=0)

            chan = np.argmin(mean_wf.min(axis=0))
            wf_chan = mean_wf[:, chan]
            sd_chan = sd_wf[:, chan]

            # --- find peak-to-peak spike width ---
            trough_idx = np.argmin(wf_chan)
            peak_idx = trough_idx + np.argmax(wf_chan[trough_idx:])

            dt_ms = 1000 / fs
            width_ms_pp = (peak_idx - trough_idx) * dt_ms
            spike_widths_ms_pp.append(width_ms_pp)

            # --- find half-width spike width ---
            # first upsample data
            U = 12
            wf_up = resample_poly(wf_chan, up=U, down=1)
            fs_up = fs * U

            # then find full width at half max
            trough_idx = np.argmin(wf_up)
            trough_val = wf_up[trough_idx]
            half_max = trough_val/2
            left = trough_idx
            while left > 0 and wf_up[left] < half_max:
                left -= 1

            right = trough_idx
            while right < len(wf_up) - 1 and wf_up[right] < half_max:
                right += 1

            hw_samples = right - left
            width_ms_hw = hw_samples * dt_ms/U
            spike_widths_ms_hw.append(width_ms_hw)

            ug = units_grp.create_group(f"unit_{int(unit_id)}")
            ug.create_dataset("mean", data=wf_chan, compression="gzip", compression_opts=4)
            ug.create_dataset("sd", data=sd_chan, compression="gzip", compression_opts=4)
            ug.create_dataset("peak_channel", data=chan)
            ug.create_dataset("n_spikes", data=len(unit_wfs))
            ug.create_dataset("spike_width_pp", data=width_ms_pp)
            ug.create_dataset("spike_width_hw", data=width_ms_hw)

    return path, spike_widths_ms_pp, spike_widths_ms_hw

if __name__ == '__main__':
    # ---- get input information ----
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--chanmap")
    parser.add_argument("--sorter")
    parser.add_argument('--destination', required=True)
    parser.add_argument("--database_path", required=True)

    args = parser.parse_args()

    # ---- copy files locally for processing ----
    copy_dir = args.data

    rec_name = Path(os.path.abspath(os.path.join(copy_dir, '..'))).name
    dest_dir = os.path.join(args.destination, rec_name, 'unfiltered')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        copy_directory_with_progress(copy_dir, dest_dir)
    else:
        print("Local copy already exists!")

    # ---- find openephys folder ----
    oe_path = os.path.join(dest_dir, os.listdir(dest_dir)[0])

    # ---- initiate sorting ----
    recording = se.read_openephys(oe_path, stream_id='0')
    channel_ids = recording.get_channel_ids()
    fs = recording.get_sampling_frequency()
    num_chan = recording.get_num_channels()
    num_seg = recording.get_num_segments()

    print("Channel ids:", channel_ids)
    print("Sampling frequency:", fs)
    print("Number of channels:", num_chan)
    print("Number of segments:", num_seg)



    my_probe = get_probe(manufacturer="cambridgeneurotech", probe_name=args.probe)
    my_probe.set_device_channel_indices(json.loads(args.chanmap))
    recording_probe = recording.set_probe(my_probe, group_mode='by_shank')
    plot_probe(recording_probe.get_probe())

    preprocess_save_folder = os.path.join(oe_path, '..', '..', 'sorting', 'preprocessed')
    recording_cmr = recording_probe
    recording_f = si.bandpass_filter(recording_probe, freq_min=300, freq_max=10000)

    removed_ch_path = os.path.join(args.data, '..', 'removed_ch.json')

    bad_channels = []
    if os.path.exists(removed_ch_path):
        with open(removed_ch_path, 'r') as f:
            j = json.load(f)
        bad_channels = np.array(j['Removed Channels']['channels'])
        print(f'Removed channels (from probe): {bad_channels}')
        ref_channels = [c for c in channel_ids if c not in bad_channels]

    if len(bad_channels) > 0:
        recording_cmr = si.common_reference(
            recording_f,
            reference='global',
            operator='median',
            ref_channel_ids=ref_channels
        )
        print('CMR filter with removed channels.')
    else:
        recording_cmr = si.common_reference(recording_f, reference='global', operator='median')

    if len(bad_channels) > 0:
        recording_final = recording_cmr.remove_channels(remove_channel_ids=bad_channels)
        print('removed channels from recording_final')
    else:
        recording_final = recording_cmr

    if not os.path.exists(preprocess_save_folder):
        recording_preprocessed = recording_final.save(format='binary', folder=preprocess_save_folder, overwrite=False)
        print("Final channel ids:", recording_preprocessed.get_channel_ids())
        print("Final number of channels:", recording_preprocessed.get_num_channels())
    else:
        print('Preprocessed data already exists, manually delete and rerun to override.')
        recording_preprocessed = si.load(preprocess_save_folder)
    print(recording_preprocessed)

    plt.savefig(os.path.join(preprocess_save_folder, 'probe_configuration.png'))
    print("Saved probe to probe_configuration.png")

    # print("")
    # print("---- Sorter Parameters ----")
    # pprint.pprint(ss.get_default_sorter_params(args.sorter))

    analyzer_folder = os.path.join(preprocess_save_folder, '..', 'analyzer_TDC_binary')
    sorting_folder = os.path.join(preprocess_save_folder, '..', 'sorting_TDC')
    if os.path.exists(sorting_folder):
        print("Loading existing sorting...")
        sorting_TDC = si.load(sorting_folder)
    else:
        print("Running sorter...")
        sorting_TDC = ss.run_sorter(
            sorter_name=args.sorter,
            recording=recording_preprocessed,
            folder=sorting_folder,
        )
    if os.path.exists(analyzer_folder):
        print("Loading existing analyzer...")
        analyzer_TDC = si.load_sorting_analyzer(analyzer_folder)
    else:
        print("Creating analyzer...")
        analyzer_TDC = si.create_sorting_analyzer(
            sorting=sorting_TDC,
            recording=recording_preprocessed,
            format='binary_folder',
            folder=analyzer_folder
        )

    print(analyzer_TDC)

    if analyzer_TDC.get_extension("random_spikes") is None:
        analyzer_TDC.compute("random_spikes")
    if analyzer_TDC.get_extension("waveforms") is None:
        analyzer_TDC.compute("waveforms")
    if analyzer_TDC.get_extension("noise_levels") is None:
        analyzer_TDC.compute("noise_levels")
    if analyzer_TDC.get_extension("templates") is None:
        analyzer_TDC.compute("templates")
    if analyzer_TDC.get_extension("spike_amplitudes") is None:
        analyzer_TDC.compute("spike_amplitudes")
    if analyzer_TDC.get_extension("spike_locations") is None:
        analyzer_TDC.compute("spike_locations")
    if analyzer_TDC.get_extension("unit_locations") is None:
        analyzer_TDC.compute("unit_locations")

    from spikeinterface_gui import run_mainwindow

    sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
    run_mainwindow(sorting_analyzer, mode='desktop', curation=True)
    test = 'FALSE'
    # ---- apply curation from GUI ----
    curation_fp = os.path.join(analyzer_folder, 'spikeinterface_gui', 'curation_data.json')
    if os.path.exists(curation_fp):
        print("Data has been manually curated.")
        curation = sc.load_curation(curation_fp)
        clean_sorting = sc.apply_curation(sorting_TDC, curation_dict_or_model=curation)
        print("Applying curation...")
        analyzer = sc.apply_curation(sorting_analyzer, curation_dict_or_model=curation)
    else:
        print("No manual curation has been found.")
        analyzer = analyzer_TDC

    # ---- run additional analysis to find quality metrics
    sqm.compute_quality_metrics(
        analyzer,
        metric_names=[
            'snr',
            'firing_rate',
            'isi_violation',
            'presence_ratio',
            'sliding_rp_violation',
            'drift',
            'amplitude_median',
            'amplitude_cv',
            'noise_cutoff'
        ],
        qm_params={"refractory_period_ms": 1}
    )
    qm = analyzer.get_extension("quality_metrics").get_data()
    print(f'There are {len(qm)} neurons!')

    # ------------------------------------------------------------------ #
    #                     Update Database                                #
    # ------------------------------------------------------------------ #
    session_id = rec_name
    sync_neurons_for_session(args.database_path, session_id, args.probe, analyzer)

    spikes_path = os.path.dirname(copy_dir)
    h5py_path = write_spike_times(spikes_path, analyzer)
    waveforms_path, spike_widths_ms_pp, spike_widths_ms_hw = write_waveforms(spikes_path, recording_preprocessed,
                                                                             analyzer.sorting)
    unit_locations_x, unit_locations_y = get_unit_locs(analyzer)
    stim_path = get_log_file_and_stimulus_windows(spikes_path)

    qm['spike_width_pp'] = spike_widths_ms_pp
    qm['spike_width_hw'] = spike_widths_ms_hw
    qm['unit_loc_x'] = np.array(unit_locations_x)
    qm['unit_loc_y'] = np.array(unit_locations_y)  # or set to array I think?

    conn = sqlite3.connect(args.database_path)
    cur = conn.cursor()

    print("Setting unit data in SQL Database...")
    for u, unit in enumerate(analyzer.sorting.get_unit_ids()):
        cur.execute("""
                    UPDATE neurons
                    SET snr                  = ?,
                        firing_rate          = ?,
                        isi_violation_ratio  = ?,
                        presence_ratio       = ?,
                        sliding_rp_violation = ?,
                        drift                = ?,
                        amplitude_median     = ?,
                        amplitude_cv         = ?,
                        noise_cutoff         = ?,
                        spike_width_pp       = ?,
                        spike_width_hw       = ?,
                        unit_loc_x           = ?,
                        unit_loc_y           = ?,

                        spike_file           = ?,
                        stimulus_file        = ?
                    WHERE session_id = ?
                      AND probe = ?
                      AND unit_id = ?
                    """, (qm.loc[unit, 'snr'], qm.loc[unit, 'firing_rate'], qm.loc[unit, 'isi_violations_ratio'],
                          qm.loc[unit, 'presence_ratio'], qm.loc[unit, 'sliding_rp_violation'],
                          qm.loc[unit, 'drift_std'],
                          qm.loc[unit, 'amplitude_median'], qm.loc[unit, 'amplitude_cv_median'],
                          qm.loc[unit, 'noise_cutoff'],
                          qm.loc[unit, 'spike_width_pp'], qm.loc[unit, 'spike_width_hw'],
                          qm.loc[unit, 'unit_loc_x'], qm.loc[unit, 'unit_loc_y'],
                          h5py_path, stim_path, session_id, args.probe, int(unit)))

    conn.commit()
    conn.close()

# ------------------------------------------------------------------ #
#                  Copy Files back to initial dir:                   #
# ------------------------------------------------------------------ #

    # ---- copy files over if the first time sorting ----
    destination_dir = os.path.join(args.data, '..', 'sorting')
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print("Copying sorting output to R:")
        copy_directory_with_progress(sorting_folder, os.path.join(destination_dir, 'sorting_TDC'))
        print("Copying curation output to R:")
        copy_directory_with_progress(analyzer_folder, os.path.join(destination_dir, 'analyzer_TDC_binary'))
        print("Success!")

    # ---- check if there are sorting differences ----
    if os.path.exists(destination_dir):
        print("Base directory already exists!")
        print("Checking for sorting output...")
        if os.path.exists(os.path.join(destination_dir, 'sorting_TDC')):
            print("Sorting output already exists!")
        else:
            copy_directory_with_progress(sorting_folder, os.path.join(destination_dir, 'sorting_TDC'))

        print("Checking for curation output...")
        if os.path.exists(os.path.join(destination_dir, 'analyzer_TDC_binary')):
            print("Curation output already exists -- checking for differences in curation.")
            gui_output_path_curr = os.path.join(destination_dir, 'analyzer_TDC_binary', 'spikeinterface_gui',
                                                'curation_data.json')
            gui_output_path_new = os.path.join(analyzer_folder, 'spikeinterface_gui', 'curation_data.json')
            if os.path.exists(gui_output_path_curr):
                if json_equal(gui_output_path_new, gui_output_path_curr):
                    print("Curation did not change!")
                else:
                    print("Curation is different - removing old file!")
                    os.remove(gui_output_path_curr)
                    shutil.copy2(gui_output_path_new, gui_output_path_curr)
            else:
                print("Analyzer folder present, but curation output missing -- replacing with new info!")
                if os.path.exists(gui_output_path_curr) and os.path.exists(gui_output_path_new):
                    os.remove(gui_output_path_curr)
                    shutil.copy2(gui_output_path_new, gui_output_path_curr)
                elif os.path.exists(gui_output_path_new) and not os.path.exists(gui_output_path_curr):
                    os.makedirs(os.path.dirname(gui_output_path_curr), exist_ok=True)
                    shutil.copy2(gui_output_path_new, gui_output_path_curr)

        else:
            print("Curation output not found -- copying to R:")
            copy_directory_with_progress(analyzer_folder, os.path.join(destination_dir, 'analyzer_TDC_binary'))

