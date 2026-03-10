from src.SiNAPSE.core.load import Database
db_path = r'C:\Users\tmerri03\Desktop\Test Sorting\test.db'
stim_lib = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
db = Database(db_path, stim_lib)
select_cols = ['unit_id', 'session_id']
conditions = {
    'manual_isi_1': ('<', 1),
    'label': ('=', 'auditory')
}
neurons = db.load_neurons_from_database(select_cols, conditions)
print(f'We have {len(neurons)} neurons.')

from src.SiNAPSE.core.sort import Recording

rec_fp = r'R:\Data\RhythmPerception\Neural Recordings\Recordings\o62g87\o62g87 Recording #5 (E1-1, HVC, 0.1AP 2.4ML-R 600um)'
samplerate = 30000
rec = Recording(rec_fp, samplerate, db=db)
log = rec.load_log_file
print(f'We have {len(log)} trials.')
rec.sort()

# from SiNAPSE.core.spikes import Neuron
# from pathlib import Path
# import os
# rec_name = Path(os.path.abspath(rec_fp)).name.strip()
# print(rec_name)
# n = Neuron(rec_name, 59, db=db, rec=rec)
# stim = n.load
# n.plot(stimuli_to_raster=["o62g87 BOS", "o62g87 REV", "ZF CON o56"], raster=True, psth=True, baseli   ne=True, padding = 0.5)
# test_save_folder = r'C:\Users\tmerri03\Desktop\Test Sorting\Test save folder'
# n.save_plots(output_path=test_save_folder, name_prefix=f'{14}_', format='svg')
#
# # print(n.get_neuron_data['spike_width_pp'])
# # print(n.get_manual_label)