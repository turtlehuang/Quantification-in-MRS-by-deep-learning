import matplotlib.pyplot as plt
import numpy as np
import os
import math
from glob import glob
import pandas as pd
import random as rdm
from tqdm import tqdm

###################Contral table################
#代謝物濃度
#baseline magnitude
#baseline linewidth
#boarden parameter
#baseline total magnitude
#noise parameter
#zero-order correct
def measure_width_hz(ppm, spectrum_data):
    if (spectrum_data.ndim == 2):
        spectrum_data = spectrum_data[:,0]
    tNAA_pos = np.where((ppm>=1.8) & (ppm<=2.2))
    tNAA_crop = spectrum_data[tNAA_pos]
    #tNAA_FWHM_val = max(tNAA_crop)*0.2
    tNAA_FWHM_val = max(tNAA_crop)*(1-0.707)
    tNAA_FWHM = np.where(tNAA_crop > tNAA_FWHM_val)
    tNAA_FWHM_width_ppm = max(ppm[tNAA_pos][tNAA_FWHM])-min(ppm[tNAA_pos][tNAA_FWHM])
    #ppm to Hz, (BW)/(total ppm = BW/B) * width_ppm = B*width_ppm
    tNAA_FWHM_width_hz = round(123.177*tNAA_FWHM_width_ppm, 3)
    return tNAA_FWHM_width_hz

#代謝物濃度
metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','PCh','Lac','mI','NAA','NAAG','Scyllo','Tau']
brain_metabos_conc_lower = np.array([0.1,1.0,4.5,1.0,1.0,3.0,6.0,0.5,0.5,0.2,4.0,7.5,0.5,0,2.0])
brain_metabos_conc_upper = np.array([1.5,2.0,10.5,2.0,2.0,6.0,12.5,2.0,2.0,1.0,9.0,17,2.5,0,6.0])
#var_brain_metabos_conc_set = rdm.uniform(brain_metabos_conc_lower,brain_metabos_conc_upper)

N_specta = 1

SHUFFLE_var_brain_metabos_conc_set = np.zeros([len(metabo_names), N_specta], dtype=np.float)
for i in range(len(metabo_names)):
    for t in range(N_specta):
        SHUFFLE_var_brain_metabos_conc_set[i,t] = rdm.uniform(brain_metabos_conc_lower[i],brain_metabos_conc_upper[i])
    rdm.shuffle(SHUFFLE_var_brain_metabos_conc_set[i])

print('SHUFFLE_var_brain_metabos_conc_set',SHUFFLE_var_brain_metabos_conc_set.shape)

AU = np.array([0.72,0.28,0.38,0.05,0.45,0.36,0.36,0.04,0.2,0.11,0.64,0.07,1])
SHUFFLE_var_AU_set = np.zeros([len(AU), N_specta], dtype=np.float)
for i in range(len(AU)):
    for t in range(N_specta):
        SHUFFLE_var_AU_set[i,t] = rdm.uniform(AU[i]*0.9,AU[i]*1.1)
    rdm.shuffle(SHUFFLE_var_AU_set[i])

print('SHUFFLE_var_AU_set',SHUFFLE_var_AU_set.shape)


FWHM=np.array([21.2,19.16,15.9,7.5,29.03,20.53,17.89,5.3,14.02,17.89,33.52,11.85,37.48])
SHUFFLE_var_FWHM_set = np.zeros([len(FWHM), N_specta], dtype=np.float)
for i in range(len(FWHM)):
    for t in range(N_specta):
        SHUFFLE_var_FWHM_set[i,t] = rdm.uniform(FWHM[i]*0.8,FWHM[i]*1.2)
    rdm.shuffle(SHUFFLE_var_FWHM_set[i])

print('SHUFFLE_var_FWHM_set',SHUFFLE_var_FWHM_set.shape)


#N_specta = 50000
for steps in tqdm(range(N_specta)):
    var_AU_set = []
    var_brain_metabos_conc_set = []
    var_FWHM_set = []
    #代謝物濃度
    metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','PCh','Lac','mI','NAA','NAAG','Scyllo','Tau']#GSH = PCh?
    var_brain_metabos_conc_set = SHUFFLE_var_brain_metabos_conc_set[:,steps]
    #print('var_brain_metabos_conc_set',var_brain_metabos_conc_set)

    #baseline magnitude
    AU = np.array([0.72,0.28,0.38,0.05,0.45,0.36,0.36,0.04,0.2,0.11,0.64,0.07,1])
    var_AU_set = SHUFFLE_var_AU_set[:,steps]
    #print('var_AU_set',var_AU_set)

    #baseline linewidth
    FWHM=np.array([21.2,19.16,15.9,7.5,29.03,20.53,17.89,5.3,14.02,17.89,33.52,11.85,37.48])
    var_FWHM_set = SHUFFLE_var_FWHM_set[:,steps]
    #print('var_FWHM_set',var_FWHM_set)

    #baseline total magnitude
    #The var_MM_amp config was in paragraph
    var_MM_amp = 0 #2500000
    #boarden parameter
    #var_boarden_t2 = rdm.randrange(200,500,50)
    var_boarden_t2 = rdm.randrange(80,350,30)
    #zero order shift parameter
    var_zero_shift = rdm.randint(-5,5)
    #AWGNoise db parameter
    #var_AWGN_db = rdm.randint(5,15)
    var_AWGN_db = rdm.randint(13,15)
    #############Load file#############
    
    working_dir = os.getcwd()
    #basis_filename = 'GAVA_press_te35_3T_test.basis'#16 kinds of metabo
    basis_filename = 'press_te30_3t_01a.basis'#15 kinds of metabo
    #basis_filename = 'gamma_press_te40_127mhz_013.basis'
    
    basis_path = os.path.join(working_dir, basis_filename)
    f_basis = open(basis_path,'r')
    BASIS = f_basis.read()
    ###############Read basis info####################
    
    SPLIT_BASIS = BASIS.split()
    indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "METABO"]
    NMUSED_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "$NMUSED"]
    conc_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "CONC"]
    ISHIFT_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "ISHIFT"]
    PPMAPP_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "PPMAPP"]
    
    BADELT = np.array(SPLIT_BASIS[SPLIT_BASIS.index('BADELT')+2].split(","))[0].astype(float)
    
    ###########################################################################
    basis_title = []
    for i in range(len(indices)):
        idx = indices[i]
        #print(SPLIT_BASIS[idx:idx+3],i)
        meta_bolite_title = SPLIT_BASIS[idx:idx+3]
        basis_title.append(meta_bolite_title)#拿每種metabo的名字
    ###########################################################################
    data_idx = NMUSED_indices
    basis_set = []
    for i in range(len(data_idx)):
        if (i <len(data_idx)-1):
            idx = data_idx[i+1]
            meta_bolite_basis = SPLIT_BASIS[idx-4096*2:idx]
            basis_set.append(meta_bolite_basis)#拿每種metabo的basis
            #print("i",i)
        else:
            idx = len(SPLIT_BASIS)
            meta_bolite_basis = SPLIT_BASIS[idx-4096*2:idx]
            #print("i final",i)        
            basis_set.append(meta_bolite_basis)
    basis_set = np.array(basis_set)#shape = Nx(sample points*2), N = number of metabo
    
    con_set = []
    for i in range(len(conc_indices)):
        idx = SPLIT_BASIS[conc_indices[i]+2]
        idx = idx.split(",")
        con_set.append(idx[0])#拿種每種metabo 的濃度 concetraition
    con_set = np.array(con_set).astype(float)
    
    ###########Config PPM############
    sample_point = int(SPLIT_BASIS[SPLIT_BASIS.index("NDATAB")+2])#NDATAB =
    BW = 1/BADELT#Hz, for GAVA_press_te35_3T_test
    B = float((SPLIT_BASIS[SPLIT_BASIS.index("HZPPPM")+2]).split(",")[0])#3T = 3*42.58= 127MHz 
    ppm_length = BW/B#16.xx ppm
    ppm_center = 4.7
    min_bound = ppm_center - (ppm_length)/2
    max_bound = ppm_center + (ppm_length)/2
    ppm = np.linspace(min_bound, max_bound, sample_point)
    ppm = ppm[::-1]#reverse
    
    ###Original####
    pos = np.where((ppm>=ppm.min()) & (ppm<=ppm.max()))
    crop_ppm = ppm[pos]
    
    decode_basis = []
    add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)#因為要累加,所以先宣告
    real_add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)
    imag_add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)
    
    ####################調整偏移量##########################
    for i in range(len(basis_set)):
        x_metabo = np.array(basis_set[i]).astype(float)
        even_x_metabo = x_metabo[::2]#Real number
        odd_x_metabo = x_metabo[1::2]#Image number
        metabo_combine = even_x_metabo + 1j*(odd_x_metabo)#Combine to complex
        fft_shift_metabo = np.fft.fftshift(metabo_combine)
        if i == 12:#NAAG
            ppmapp_range = np.where((ppm>float((SPLIT_BASIS[PPMAPP_indices[i-1]+3]).split(",")[0])) &(ppm<=float(SPLIT_BASIS[PPMAPP_indices[i-1]+2])))[0]
        else:
            ppmapp_range = np.where((ppm>float((SPLIT_BASIS[PPMAPP_indices[i]+3]).split(",")[0])) &(ppm<=float(SPLIT_BASIS[PPMAPP_indices[i]+2])))[0]
        ppmapp_range_max = fft_shift_metabo[ppmapp_range].argmax()# 找實際 0 ppm peak的位置
        #plt.plot(ppm[ppmapp_range],fft_shift_metabo[ppmapp_range])
        ppm_zero_pos = np.where((ppm<=0.01)&(ppm>=-0.01))[0][0]#找 0 ppm 的位置
        #ppmapp_shift = abs(ppmapp_range[ppmapp_range_max] - ppm_zero_pos)
        ppmapp_shift = (ppmapp_range[ppmapp_range_max] - ppm_zero_pos)
        #print(f'ppmapp_shift: {ppmapp_shift}')
        fft_shift_ISHIFT_metabo = np.roll(fft_shift_metabo,-ppmapp_shift)#根據BASIS中的資訊去SHIFT來校正
        crop_shift_metabo = fft_shift_ISHIFT_metabo[pos]#Crop特定ppm的頻譜
        real_add_crop_shift_metabo += crop_shift_metabo.real
        imag_add_crop_shift_metabo += crop_shift_metabo.imag
        decode_basis.append(crop_shift_metabo)
    
    decode_basis = np.array(decode_basis)
    cb_add_crop_shift_metabo = real_add_crop_shift_metabo + 1j*imag_add_crop_shift_metabo
    ############ Consider concertration################

    brain_betabo_conc_table = {
            "names": metabo_names,
            "conc": var_brain_metabos_conc_set
            }
    brain_betabo_conc_table_df = pd.DataFrame(brain_betabo_conc_table)
    
    #8 = PCh
    
    add_conc_decode_basis = np.zeros(decode_basis[0].size).astype(float)
    real_add_conc_decode_basis = np.zeros(decode_basis[0].size).astype(float)
    imag_add_conc_decode_basis = np.zeros(decode_basis[0].size).astype(float)
    
    #plt.figure()
    unit_decode_basis_set = np.zeros([len(decode_basis)-2,len(decode_basis[0])],dtype=np.csingle)
    #plt.figure(figsize=(15,10))
    for i in range(len(decode_basis)-2):
        #print('basis_title[i][2]',basis_title[i][2])
        unit_decode_basis = decode_basis[i]/con_set[i]
        unit_decode_basis_set[i] = unit_decode_basis
        conc_decode_basis = unit_decode_basis*brain_betabo_conc_table_df['conc'][i]
        #plt.figure()
        #plt.title(str(basis_title[i][2]))
        #plt.plot(crop_ppm,conc_decode_basis[pos], label=str(basis_title[i][2]))
        #plt.xlim(4.5,0.5)
        #plt.ylim(-1000,50000000)
        #plt.legend(loc='upper right')
        real_add_conc_decode_basis += conc_decode_basis.real
        imag_add_conc_decode_basis += conc_decode_basis.imag
    #############答案, netabolite only spectrum#############
    add_conc_decode_basis = real_add_conc_decode_basis + 1j*imag_add_conc_decode_basis

    
    ##################MM Baseline###################
    def g(x, A, μ, σ):
        return (A / (σ * math.sqrt(2 * math.pi))) * np.exp(-(x-μ)**2 / (2*σ**2))
    
    group_name = ['MM09','MM12','MM14','MM16','MM20','MM21','MM23','MM26','MM30','MM31','MM37','MM38','MM40']
    AU = var_AU_set
    freq = ['0.9','1.21','1.38','1.63','2.01','2.09','2.25','2.61','2.96','3.11','3.67','3.8','3.96']
    FWHM = var_FWHM_set
    MM_table = {
            "name": group_name,
            "AU":AU,
            "freq":freq,
            "FWHM":FWHM
            }
    MM_table_df = pd.DataFrame(MM_table)
    
    #plt.figure(figsize=(10,5))
    #var_MM_amp = rdm.randrange(200000*3,400000*3,30000) #2500000
    max_pure_metabo_peak = int(max(add_conc_decode_basis.real))
    #var_MM_amp = rdm.randrange(int(max_pure_metabo_peak*0.002), int(max_pure_metabo_peak*0.004), int(max_pure_metabo_peak*0.001))
    var_MM_amp = rdm.randrange(int(max_pure_metabo_peak*0.001), int(max_pure_metabo_peak*0.003), int(max_pure_metabo_peak*0.001))
    add_MM = np.zeros(crop_ppm.size).astype(float)
    for i in range(len(MM_table_df)):
        amp = float(MM_table_df['AU'][i])*var_MM_amp#2500000
        fre_ppm = float(MM_table_df['freq'][i])
        w = (float(MM_table_df['FWHM'][i])/B)/2.355#FWHM to sigma
        mmm = g(crop_ppm,amp,fre_ppm,w)
        #print("mmm",mmm.real)
        #plt.figure()
        '''
        plt.title(str(MM_table_df.loc[i]))
        plt.plot(crop_ppm,mmm.real)
        plt.xlim((4.5,0.5))
        '''
        add_MM += mmm.real
    
    component_sum = add_conc_decode_basis+add_MM#2500000
    '''
    plt.figure(figsize=(10,5))
    plt.title('MM + conc')
    plt.plot(crop_ppm,component_sum[pos])
    plt.xlim((4.2,0.5))
    plt.ylim((0,0.7*10**8))
    '''
    
    #################Boarden linewith###############
    '''
    plt.figure(figsize=(15,10))
    plt.subplot(211)
    plt.title(f'Before boardening, linewidth: {measure_width_hz(crop_ppm, component_sum[pos])} Hz')
    plt.plot(crop_ppm, component_sum[pos])
    plt.xlim((4.2,0.5))
    '''
    filted_tdata_ori = np.fft.ifft(component_sum)
    
    x_t = np.arange(0,len(filted_tdata_ori))
    # 個案測試
    #var_boarden_t2_list = [300,300,1000,1000]
    
    #var_boarden_t2 = var_boarden_t2_list[steps]
    #var_boarden_t2 = 150
    exp_adop_filt = np.exp(-(x_t/var_boarden_t2))
    filted_tdata = filted_tdata_ori*exp_adop_filt
    #################Zero order correct###############    
    #zero_shift_filter = np.exp(var_zero_shift)    

    ##################Crop period################
    filted_sdata = np.fft.fft(filted_tdata)

    #pos = np.where((ppm>=0.5) & (ppm<=4.5))
    pos = np.where((ppm>=0.5) & (ppm<=4.2))
    crop_ppm = ppm[pos]

    crop_filted_sdata = filted_sdata[pos]
    pure_metabo_basis = add_conc_decode_basis[pos]
    crop_MM = add_MM[pos]
    
    ##############base basis set
    base_basis_set = np.zeros([len(unit_decode_basis_set),len(unit_decode_basis_set[0][pos])])
    base_basis_set = np.array([cont[pos] for i,cont in enumerate(unit_decode_basis_set)])
    
    '''
    plt.subplot(212)
    #plt.title('Boarden T2 = {}'.format(var_boarden_t2))
    plt.title(f'Boarden T2 = {var_boarden_t2}, linewidth:{measure_width_hz(crop_ppm, crop_filted_sdata)} Hz')
    plt.plot(crop_ppm, crop_filted_sdata)
    plt.xlim((4.2,0.5))
    '''
    '''
    plt.figure(figsize=(15,10))
    plt.subplot(211)
    plt.plot(crop_ppm, crop_filted_sdata)
    plt.xlim(4.2,0.5)
    '''
    ##################Add noise##################
    #https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    
    crop_filted_sdata_watts = crop_filted_sdata ** 2
    crop_filted_sdata_db = 10 * np.log10(crop_filted_sdata_watts)
    # Additive White Gaussian Noise (AWGN)
    # Set a target SNR
    # 個案測試
    #var_AWGN_db_list = [20,10,20,10]
    #var_AWGN_db = var_AWGN_db_list[steps]
    #steps = 0
    #var_AWGN_db = 15
    
    target_snr_db = var_AWGN_db
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(crop_filted_sdata_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(crop_filted_sdata_watts))
    # Noise up the original signal
    y_volts = crop_filted_sdata + noise_volts
    add_noise_crop_filted_sdata = y_volts
        
    '''
    in_mean = np.mean(add_noise_crop_filted_sdata.real)
    in_std = np.std(add_noise_crop_filted_sdata.real)
    pure_metabo_basis = ((pure_metabo_basis.real - in_mean)/in_std) +1j*((pure_metabo_basis.imag - in_mean)/in_std)
    add_noise_crop_filted_sdata = ((add_noise_crop_filted_sdata.real - in_mean)/in_std) +1j*((add_noise_crop_filted_sdata.imag - in_mean)/in_std)
    '''
    '''
    plt.subplot(212)
    plt.title('SNR = {}'.format(var_AWGN_db))
    plt.plot(crop_ppm, add_noise_crop_filted_sdata)
    plt.xlim(4.2,0.5)
    '''
    ##################View result##################
    
    if steps < 5:
        #print('pure metabot peak AU:', max(pure_metabo_basis.real))
        #print('var_MM_amp ',var_MM_amp)
        #print('var_MM_amp/max(pure_metabo_basis.real)',var_MM_amp/max(pure_metabo_basis.real))
        plt.figure(figsize=(15,10))
        plt.subplot(3,1,1)
        plt.title('Pure metabolite spectrum data')
        plt.plot(crop_ppm,pure_metabo_basis)
        #plt.xlim((4.5,0.5))
        plt.xlim((4.2,0.5))
        plt.ylim(0,0.4*10**8)
        
        plt.subplot(3,1,2)
        plt.title('Gererated MM')
        plt.plot(crop_ppm,crop_MM)
        #plt.xlim((4.5,0.5))
        plt.xlim((4.2,0.5))
        plt.ylim(0,0.4*10**8)
        #plt.ylim(0,0.4*10**8)
        
        plt.subplot(3,1,3)
        plt.title(f'Add artifacts: Broaden T2 = {var_boarden_t2} AWGN_db ={var_AWGN_db} ')
        plt.plot(crop_ppm, add_noise_crop_filted_sdata)
        plt.xlim((4.2,0.5))
        #plt.xlim((4.5,0.5))
        plt.ylim((0,0.4*10**8))
        
        '''
        sav_filename = f'gen_{steps}.png'
        #print(f"save: f{sav_filename}")
        sav_filename_path = os.path.join(working_dir,'gen_folder', sav_filename)    
        plt.savefig(sav_filename)#Why I can't save it into sub folder
        '''
    
    np.savez(os.path.join(working_dir,'gen_folder', f'generate_basis_{steps}'), X=add_noise_crop_filted_sdata, Y=pure_metabo_basis, ppm = crop_ppm)
    np.savez(os.path.join(working_dir,'gen_folder','other_parameters' ,f'other_parameters_{steps}'), var_MM_amp = var_MM_amp, var_boarden_t2 = var_boarden_t2, var_zero_shift= var_zero_shift, var_AWGN_db = var_AWGN_db)
    brain_betabo_conc_table_df.to_pickle(os.path.join(working_dir,'gen_folder','brain_betabo_conc_table_df',f'brain_betabo_conc_table_df_{steps}'))
    MM_table_df.to_pickle(os.path.join(working_dir,'gen_folder','MM_table_df',f'MM_table_df_{steps}'))    
    if steps == 0:
        #np.savez(os.path.join(working_dir,'base_basis_set'),data = base_basis_set)
        np.savez(os.path.join(working_dir,'42ppm_base_basis_set'),data = base_basis_set)
    #base_basis_set
    print("Current steps: ", steps)
