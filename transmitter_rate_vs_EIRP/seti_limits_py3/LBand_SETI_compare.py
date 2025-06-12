from ET_power_law import *

import matplotlib.pylab as plt


def lband_compare(max_distance, snr, save=False):
    # ---------------------------
    # Edit values below  vvvvvv
    # ---------------------------
    # Your new values ( e.g. Enriquez 2017  x 100)

    project = 'This Study: L-Band'  # Project name
    telescope = 'GBT'     # Telescope name
    N_stars = 5.84e12    # Estimated number of stars
    band = 660e6      # Total bandwidth [Hz]
    central_freq = 1.5e9  # Central bandwidth [Hz]

    # Telescope diameter meters (single dish in current version)
    dish_diam = 100
    dish_Tsys = 20  # Telescope Tsys [Kelvin], taken from GBT observers guide
    dish_app_eff = 0.7  # Telescope Apperture Efficiency

    SNR_threshold = snr  # Survey threshold   [sigma above the mean]
    spectral_resolution = 2.79  # Spectral resolution [Hz]
    scan_obs_time = 300  # Observation time per scan [sec]
    # max_distance = 505.33  # Maximum distance  [pc]
    max_drift = 5  # [Hz/sec]
    beta = 3 / (max_drift * 18)  # Dimensionless, [Hz / (Hz/sec * sec)]

    iband = 800e6  # Instantaneous Bandwidth [Hz]

    shape = '*'    # Figure shape
    color = 'r'  # Figure color
    y_label_units = True  # Units in Y label

    # ---------------------------
    # Edit values above  ^^^^^
    # ---------------------------
    # Calculating limits

    zeta_AO = 1e3*0.5 / 1e13

    freq_range_norm = (band/central_freq)
    SEFD = calc_SEFD(calc_DishArea(dish_diam), dish_Tsys,
                     eff=dish_app_eff)  # 10 Jy (GBT)

    # SEFD = 7.7251

    Sens = calc_Sensitivity(
        SNR_threshold, spectral_resolution, scan_obs_time, SEFD=SEFD)

    Sens_drift = Sens/beta

    dist_m = (max_distance*3.26156 * u.lyr.to('m'))

    EIRP = calc_EIRP_min(dist_m, Sens)
    EIRP_drift = calc_EIRP_min(dist_m, Sens_drift)
    #EIRP = 1926e12
    #EIRP = 491.6e12

    survey_rarity = N_stars*freq_range_norm
    survey_speed = SEFD**2*spectral_resolution/iband
    survey_sky = 97 * calc_BeamSize(dish_diam, central_freq)
    survey_DFM = survey_sky * band / Sens**(3/2.)

    def print_project():
        print('~o~', project, ' (', telescope, ') ', '~o~')
        print('SEFD :', SEFD)
        print('Sens :', Sens)
        print('Max Distance (m) :', dist_m)
        print('EIRP :', EIRP)
        print('EIRP (max drift rate) :', EIRP_drift)
        print('BeamSize :', calc_BeamSize(dish_diam, central_freq))
        print('Sky Coverage :', survey_sky)
        print('CWTFM :',  zeta_AO * (EIRP) / (survey_rarity))
        print('DFM :', survey_DFM)

    print_project()

    # ---------------------------
    # Comparing SETI limits

    if save:
        compare_SETI_limits(EIRP, survey_rarity, shape=shape, color=color,
                            project=project, y_label_units=y_label_units, save_as='LBand_seti_compare')

    lband_dict = {'EIRP': EIRP, 'EIRP_drift': EIRP_drift, 'rarity': survey_rarity, 'shape': shape,
                  'color': color, 'project': project, 'y_label_units': y_label_units}

    return lband_dict
