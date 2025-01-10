import os
import sys

import numpy as np
import pandas as pd
import setigen as stg
from astropy import units as u
from alive_progress import alive_bar
from turbo_seti.find_doppler.find_doppler import FindDoppler
import math

from blimpy import Waterfall

# A redux of Krishnakumar Bhattaram's signal insertion script by Carmen Choza, with help from Bryan Brzycki.

# The output name for inserted files.
outfilename = "drifting_centered_1000_10p_2w_square_HIP95802"

out_directory = '/datax/scratch/cgchoza/testpipes/TIC159107668/injected'       # Output directory.

# File for signal insertion.
# file_path = '/datag/pipeline/AGBT21A_996_49/blc03/blc03_guppi_59411_54700_UGCA127_0095.rawspec.0000.h5'
file_path = '/datag/blpd1/datax2/kepler160_092020/blc43_guppi_59103_04780_DIAG_HIP95802_0019.rawspec.0000.fil'
#file_path = '/datax/scratch/cgchoza/spliced_blc00010203040506o7o0111213141516o7o021222324252627_guppi_58810_21241_And_XIV_0058.gpuspec.0000.h5'

verbose = True                      # Parameter to control print statements.
# Parameter to control .dat and .h5 and .log outputs from iterative step.
save_all_outputs = True

# Parameter to control whether signals are injected at the edge or center of a fine frequency channel.
edge_centered = False


def waterfall_generator(waterfall_fn, fchans=1048576, f_begin=None, f_end=None, f_shift=1048576):
    """
    Adapted from setigen's split_waterfall_generator method. This method takes five parameters:

        waterfall_fn: string, the filepath to the file to be used for signal insertion.
        fchans: int, a number of fine channels to define the width of a signal insertion bloc. 
                Defaults to the width of one coarse channel.
        f_begin: int, a fine channel indicating where to begin insertion.
        f_end: int, a fine channel indicating where to begin insertion.
        f_shift: int, a number of fine channels to move before beginning a new signal insertion bloc.

    The method iteratively loads blocs of the spectrogram with width = fchans and returns a generator 
    that yields Waterfall objects when invoked. 

    """

    info_wf = Waterfall(waterfall_fn, load_data=False)
    fch1 = info_wf.header['fch1']
    nchans = info_wf.header['nchans']
    df = abs(info_wf.header['foff'])
    tchans = info_wf.container.selection_shape[0]

    if f_end is None or f_end > fch1:     # If no end frequency is specified or f_end is outside of frequency bounds,
        # then reset f_stop and f_end to the first channel.
        f_stop = fch1
        f_end = fch1
    else:
        f_stop = f_end

    # If no start frequency is specified or outside bounds,
    if f_begin is None or f_begin < fch1 - nchans * df:
        # then reset f_start and f_begin to the last fine channel.
        f_start = fch1 - fchans * df
        f_begin = fch1 - nchans * df
    else:
        f_start = f_end - fchans*df

    if f_shift == None:                   # If no shift is specified, shift one width.
        f_shift = fchans

    # Creates a generator for use to iterate down frequency blocs, starting from highest frequency.

    if verbose:
        print("\nStarting at f_begin = ", f_begin)
        print("Ending at f_end = ", f_end)
        print("Iterating with spacing = ", f_shift * df, '\n')
    while f_start >= f_begin:
        waterfall = Waterfall(waterfall_fn,
                              f_start=f_start,
                              f_stop=f_stop,
                              t_start=0,
                              t_stop=tchans)

        yield waterfall

        f_start -= f_shift * df
        f_stop -= f_shift * df


def insert_signal(frame, freq, drift, snr, width):
    '''
    This method handles signal insertion. It takes five parameters:

        frame: a Setigen Frame object, yielded from the generator produced by the 
               waterfall_generator method.
        freq: float, the frequency of the signal to be inserted.
        drift: float, the drift of the signal to be inserted.
        snr: int, the snr of the signal to be inserted.
        width: the bandwidth of the signal to be inserted.

    '''
    doppler_smearing = True

    if verbose:
        print("Inserting signal", freq, drift, snr, width, frame.df,
              frame.dt, frame.fch1, frame.tchans, frame.fchans)
        print(f"Standard deviation is: {frame.noise_std}")
        print(f"Mean is: {frame.noise_mean}")

    if drift == 0:
        doppler_smearing = False

    signal = frame.add_signal(path=stg.constant_path(f_start=freq*u.MHz + 0*frame.df*u.Hz, drift_rate=drift*u.Hz/u.s),
                              t_profile=stg.constant_t_profile(
                                  frame.get_intensity(snr=snr)),
                              f_profile=stg.box_f_profile(
                                  width=width*frame.df),
                              bp_profile=stg.constant_bp_profile(level=1),
                              doppler_smearing=doppler_smearing,
                              smearing_subsamples=15)


def save_frame(outfilename, frame, before=True):
    # Save out an hdf5 file.

    if save_all_outputs:
        outname = str(int(frame.fch1)) + '_' + outfilename
    else:
        outname = outfilename

    if before:
        frame.save_hdf5(filename=out_directory + '/' + outname + '.h5')
    else:
        frame.save_hdf5(filename=out_directory +
                        '/' + outname + '_inserted.h5')


def load_and_extract(outfilename, before=True):
    # Load in the .dat file for the bloc.
    # If before signal insertion, load .dat as normal; if after, load inserted .dat.
    if before:
        f = open(out_directory + '/' + outfilename + '.dat', 'r')
    else:
        f = open(out_directory + '/' + outfilename + '_inserted.dat', 'r')

    try:
        # Isolates hit lines.
        data = [dataline.split() for dataline in [
            line for line in f.readlines() if line[0] != '#']]
        # Extract information.
        data = [[float(val) for val in dataline] for dataline in data]
    except:
        data = []

    if len(data) == 0:
        data = [[0.0]*12]
        num_hits = 0
    else:
        num_hits = len(data)

    f.close()

    return num_hits, data


def clean_outputs(outfilename, save_all_outputs):
    # If a file exists from a previous frame, remove it from the output directory.
    if save_all_outputs:
        try:
            os.remove(out_directory + '/' + outfilename + '.dat')
            os.remove(out_directory + '/' + outfilename + '.log')
        except:
            pass
    else:
        try:
            os.remove(out_directory + '/' + outfilename + '.dat')
            os.remove(out_directory + '/' + outfilename + '.log')
            os.remove(out_directory + '/' + outfilename + '_inserted.dat')
            os.remove(out_directory + '/' + outfilename + '_inserted.log')
        except:
            pass


def turbo_runner(waterfall_gen, drifts=None, snrs=None, widths=None, max_drift=4, min_drift=0.00001, min_snr=10, num_inserted=1, num_frames=None):
    '''
    The main method that drives file loading, iteration, signal injection, and output saving.
    It accepts nine parameters:

        waterfall_gen: the generator produced by waterfall_generator, for iterating over the input file.
        drifts: list of floats with length of the number of signals to insert, the drifts to insert.
        snrs: list of floats with length the number of signals to insert, the SNRs to insert.
        widths: list of floats with length the number of signals to insert, the bandwidths to insert.
        max_drift: int, the maximum drift boundary for turboSETI to search for signals with the FindDoppler method.
        min_snr: int, the minimum drift boundary for turboSETI to search with the FindDoppler method.
        num_inserted: int, the number of signals to insert.
        num_frames: int, the number of frames in waterfall_gen, calculated from fchans and the total channels of the input file.

    Returns:

        pre_turbo: list of turboSETI hits pre-signal insertion.
        sig_inserted: list of injected signals.
        post_turbo: list of turboSETI hits post-signal insertion.

    '''

    # A list to hold turboSETI hit results, before any signal insertion.
    pre_turbo = np.zeros([0, 12])
    # A list to hold turboSETI hit results, after signal insertion.
    post_turbo = np.zeros([0, 12])
    # A list to hold inserted signals.
    sig_inserted = np.zeros([0, 4])

    with alive_bar(num_frames * num_inserted) as bar:

        for waterfall in waterfall_gen:

            # Run turboSETI before any signal injection to isolate existing hits.

            frame = stg.Frame(waterfall=waterfall)
            save_frame(outfilename, frame, before=True)

            # NOTE: turboSETI accepts and converts filterbank files, but this is useful if outputs are desired for inspection with blimpy,
            # and avoids an h5py bug on the UC Berkeley compute nodes.

            if save_all_outputs:
                outname = str(int(frame.fch1)) + '_' + outfilename
            else:
                outname = outfilename

            h5_fn = out_directory + '/' + outname + '.h5'

            find_seti_event = FindDoppler(h5_fn,
                                          max_drift=max_drift, min_drift=min_drift, snr=min_snr, out_dir=out_directory, gpu_backend=True, gpu_id=1)
            find_seti_event.search()

            pre_hits, pre_data = load_and_extract(outname, before=True)

            clean_outputs(outfilename, save_all_outputs)

            # Insert signals

            # Extract frequency information. NOTE: If many signals are chosen for a narrow window, they may overlap.
            if num_inserted == 1:
                if edge_centered:
                    freq = frame.get_frequency(frame.fchans//4)*u.Hz/u.MHz
                else:
                    freq = frame.get_frequency(
                        frame.fchans//4)*u.Hz/u.MHz + (frame.df/2)*u.Hz/u.MHz
            else:
                inchans = np.linspace(
                    0, frame.fchans, num_inserted + 2, dtype=np.int32)
                inchans = inchans[1:-1]
                if edge_centered:
                    freq = [frame.get_frequency(
                        chan)*u.Hz/u.MHz + (frame.df/2)*u.Hz/u.MHz for chan in inchans]
                else:
                    freq = [frame.get_frequency(
                        chan)*u.Hz/u.MHz for chan in inchans]

            # If drifts, snrs, and widths not given, set defaults. If given, take input lists.
            if drifts is None:
                drift = 0
            else:
                drift = list(next(drifts))

            if snrs is None:
                snr = 40
            else:
                snr = list(next(snrs))

            if widths is None:
                width = 3
            else:
                width = list(next(widths))

            # Insert num_inserted signals with given parameter lists.

            siginsert = []
            if num_inserted == 1:
                insert_signal(frame, freq, drift[0], snr[0], width[0])
                siginsert.append([freq/1e6, drift, snr, width])
                bar()
            else:
                for f, d, s, w in zip(freq, drift, snr, width):
                    insert_signal(frame, f, d, s, w)
                    siginsert.append([f/1e6, d, s, w])
                    bar()

            # Run FindDoppler on the bloc with injected signals.

            save_frame(outfilename, frame, before=False)

            h5_fn = out_directory + '/' + outname + '_inserted.h5'

            find_seti_event = FindDoppler(h5_fn,
                                          max_drift=max_drift, min_drift=min_drift, snr=min_snr, out_dir=out_directory, gpu_backend=True, gpu_id=1)
            find_seti_event.search()

            # Load in the .dat file for the bloc.
            post_hits, post_data = load_and_extract(outname, before=False)

            clean_outputs(outfilename, save_all_outputs)

            pre_turbo = np.concatenate((pre_turbo, pre_data.copy()), axis=0)
            sig_inserted = np.concatenate(
                (sig_inserted, siginsert.copy()), axis=0)
            post_turbo = np.concatenate((post_turbo, post_data.copy()), axis=0)

            if verbose:
                print("Frequency range: " + str(frame.fch1 - frame.fchans * frame.df) + " to " +
                      str(frame.fch1))
                print("Inserting at: ", freq)
                print("Number of hits before insertion: " + str(pre_hits))
                print("Number of signals inserted: " + str(num_inserted))
                print("Number of hits after insertion: " + str(post_hits))

    return pre_turbo, sig_inserted, post_turbo


def main():

    # NOTE: turboSETI does not accept a file smaller than one coarse channel's worth of data.
    fchans = 1048576
    f_begin = None
    f_shift = 1048576
    f_end = None

    info_wf = Waterfall(file_path, load_data=False)
    chans = info_wf.header['nchans']

    num_to_insert = 10

    num_frames = int(chans/fchans)
    print(f"Number of coarse channels/frames: {num_frames}")

    # with open('/datax/scratch/cgchoza/turbo_debug_output.txt') as f:

    #     turbo_rates = [float(*dataline.split(':')[1:]) for dataline in [
    #         line for line in f.readlines() if "Start searching for hits" in line]]
    #     print(turbo_rates)

    # Insert signals in a range from -5 to 5 across the frequency range
    driftlist = np.linspace(-5, 5, num=int(num_to_insert*num_frames))
    # driftlist = turbo_rates
    drifts = (driftlist[num_to_insert*N:num_to_insert*(N+1)][::-1]
              for N in range(int(num_to_insert*chans/fchans)))
    # drifts = ([0.0 for _ in range(num_to_insert)]
    #           for _ in range(int(chans/fchans)))

    # Bandwidth of signal in units of fine channel widths
    widths = ([2.0 for _ in range(num_to_insert)]
              for _ in range(int(num_frames)))

    # Insert all signals at 1000 SNR
    snrs = ([1000.0 for _ in range(num_to_insert)]
            for _ in range(int(num_frames)))
    # snrlist = np.linspace(10, 650, num=int(num_to_insert*chans/fchans))
    # snrs = (snrlist[num_to_insert*N:num_to_insert*(N+1)]
    #         for N in range(int(num_to_insert*chans/fchans)))

    print("Beginning signal insertion...")

    waterfall_gen = waterfall_generator(file_path,
                                        fchans,
                                        f_begin=f_begin,
                                        f_end=f_end,
                                        f_shift=f_shift)

    pre_turbo, signals_inserted, post_turbo = turbo_runner(
        waterfall_gen, drifts=drifts, snrs=snrs, widths=widths, min_drift=0.00001, max_drift=4, min_snr=10, num_inserted=num_to_insert, num_frames=num_frames)

    # Convert to pandas DataFrame and save out signal injection and recovery information.

    print("Complete! Saving outputs...")

    pre_turbo = pd.DataFrame(pre_turbo, columns=['Top_Hit_#', 'Drift_Rate', 'SNR', 'Uncorrected_Frequency', 'Corrected_Frequency',
                                                 'Index', 'freq_start', 'freq_end', 'SEFD', 'SEFD_freq', 'Coarse_Channel_Number', 'Full_number_of_hits'])
    signals_inserted = pd.DataFrame(signals_inserted, columns=[
                                    'Frequency', 'Drift Rate', 'SNR', 'Bandwidth'])
    post_turbo = pd.DataFrame(post_turbo, columns=['Top_Hit_#', 'Drift_Rate', 'SNR', 'Uncorrected_Frequency', 'Corrected_Frequency',
                                                   'Index', 'freq_start', 'freq_end', 'SEFD', 'SEFD_freq', 'Coarse_Channel_Number', 'Full_number_of_hits'])

    pre_turbo.to_csv(f"{out_directory}/pre_insertion_hits_{outfilename}.csv")
    signals_inserted.to_csv(
        f"{out_directory}/signals_inserted_{outfilename}.csv")
    post_turbo.to_csv(f"{out_directory}/post_insertion_hits_{outfilename}.csv")

    print("Complete.")


if __name__ == '__main__':
    sys.exit(main())
