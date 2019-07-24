import matplotlib.pyplot as plt
import numpy as np
import os
import math
from glob import glob
import pandas as pd

#############Load file#############

working_dir = os.getcwd()
#basis_filename = 'GAVA_press_te35_3T_test.basis'
basis_filename = 'press_te30_3t_01a.basis'#Something wrong
#basis_filename = 'basis_15ms_3T.basis'

basis_path = os.path.join(working_dir, basis_filename)
f_basis = open(basis_path,'r')
BASIS = f_basis.read()
SPLIT_BASIS = BASIS.split()
indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "METABO"]
NMUSED_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "$NMUSED"]
conc_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "CONC"]
ISHIFT_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "ISHIFT"]
PPMAPP_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "PPMAPP"]

BADELT = np.array(SPLIT_BASIS[SPLIT_BASIS.index('BADELT')+2].split(","))[0].astype(float)


basis_title = []
for i in range(len(indices)):
    idx = indices[i]
    print(SPLIT_BASIS[idx:idx+3],i)
    meta_bolite_title = SPLIT_BASIS[idx:idx+3]
    basis_title.append(meta_bolite_title)#拿每種metabo的名字

data_idx = NMUSED_indices
basis_set = []
for i in range(len(data_idx)):
    if (i <len(data_idx)-1):
        idx = data_idx[i+1]
        meta_bolite_basis = SPLIT_BASIS[idx-4096*2:idx]
        basis_set.append(meta_bolite_basis)#拿每種metabo的basis
        print("i",i)
    else:
        idx = len(SPLIT_BASIS)
        meta_bolite_basis = SPLIT_BASIS[idx-4096*2:idx]
        print("i final",i)        
        basis_set.append(meta_bolite_basis)
basis_set = np.array(basis_set)


con_set = []
for i in range(len(conc_indices)):
    idx = SPLIT_BASIS[conc_indices[i]+2]
    idx = idx.split(",")
    con_set.append(idx[0])#拿種每種metabo 的濃度 concetraition
con_set = np.array(con_set).astype(float)

###########Config PPM############
sample_point = 4096
BW = 1/BADELT#Hz, for GAVA_press_te35_3T_test
B = 127.75#3T = 3*42.58= 127MHz
ppm_length = BW/B#16.xx ppm
ppm_center = 4.7
min_bound = ppm_center - (ppm_length)/2
max_bound = ppm_center + (ppm_length)/2
ppm = np.linspace(min_bound, max_bound, sample_point)
ppm = ppm[::-1]#reverse

pos=np.where((ppm>=-0.5) & (ppm<4.5))#pos = position index
crop_ppm = ppm[pos]


decode_basis = []
add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)#因為要累加,所以先宣告
real_add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)
imag_add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)
#cb_add_crop_shift_metabo = np.zeros(crop_ppm.size).astype(float)
for i in range(len(basis_set)):
#    if ( (i==2) | (i==6)|(i==7)|(i==10)|(i==11) ):
        x_metabo = np.array(basis_set[i]).astype(float)
        even_x_metabo = x_metabo[::2]#Real number
        odd_x_metabo = x_metabo[1::2]#Image number
        metabo_combine = even_x_metabo + 1j*(odd_x_metabo)#Combine to complex
        fft_shift_metabo = np.fft.fftshift(metabo_combine)
        IFFT = int(SPLIT_BASIS[ISHIFT_indices[i]+2])
        ppmapp_range = np.where((ppm>float((SPLIT_BASIS[PPMAPP_indices[i]+3]).split(",")[0])) &(ppm<=float(SPLIT_BASIS[PPMAPP_indices[i]+2])))[0]
        ppmapp_range_max = fft_shift_metabo[ppmapp_range].argmax()
        ppm_zero_pos = np.where((ppm<=0.01)&(ppm>=-0.01))[0][0]
        ppmapp_shift = abs(ppmapp_range[ppmapp_range_max] - ppm_zero_pos)
        '''
        plt.figure()
        plt.title(basis_title[i])
        plt.plot(fft_shift_metabo[ppmapp_range])
        '''
        fft_shift_ISHIFT_metabo = np.roll(fft_shift_metabo,-ppmapp_shift)#根據BASIS中的資訊去SHIFT來校正
        print(f'{basis_title[i]} ppmapp_shift: {ppmapp_shift}, IFFT: {IFFT}')
        '''
        plt.figure()
        plt.title(basis_title[i])
        plt.plot(ppm,fft_shift_ISHIFT_metabo)
        '''
#        crop_shift_metabo = fft_shift_ISHIFT_metabo[pos]#Crop特定ppm的頻譜
        crop_shift_metabo = fft_shift_ISHIFT_metabo[pos]#Crop特定ppm的頻譜
#        cb_add_crop_shift_metabo += crop_shift_metabo
        real_add_crop_shift_metabo += crop_shift_metabo.real
        imag_add_crop_shift_metabo += crop_shift_metabo.imag
        decode_basis.append(crop_shift_metabo)
decode_basis = np.array(decode_basis)
cb_add_crop_shift_metabo = real_add_crop_shift_metabo + 1j*imag_add_crop_shift_metabo

for i in range(len(decode_basis)):    
    plt.figure(figsize=(15,5))
    plt.title(basis_title[i])
    plt.plot(crop_ppm, decode_basis[i])
    plt.xlim((4.5,-0.5))

plt.figure(figsize=(10,5))
plt.title('cb_add_crop_shift_metabo')#全部metabo的basis的實部累加與虛部累加
plt.plot(crop_ppm,cb_add_crop_shift_metabo)
plt.xlim((4.5,0.5))

plt.figure(figsize=(10,5))
plt.title('abs_add_crop_shift_metabo')
plt.plot(crop_ppm,abs(cb_add_crop_shift_metabo))
plt.xlim((4.5,0.5))

############ Consider concertration################
metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','GSH','Lac','mI','NAA','NAAG','PC','PCr','PE','Tau','Glx','tCho','tCr','tNAA']
brain_metabos_conc=np.array([0.8,1.5,7.5,1.5,1.5,4.5,9.25,1.25,2.25,0.6,6.5,12.25,1.5,0,0,0,0,0,0,0,0])
brain_betabo_conc_table = {
        "names": metabo_names,
        "conc": brain_metabos_conc
        }
brain_betabo_conc_table_df = pd.DataFrame(brain_betabo_conc_table)

#8 = PCh

add_conc_decode_basis = np.zeros(decode_basis[0].size).astype(float)
real_add_conc_decode_basis = np.zeros(decode_basis[0].size).astype(float)
imag_add_conc_decode_basis = np.zeros(decode_basis[0].size).astype(float)

for i in range(len(decode_basis)-2):
        unit_decode_basis = decode_basis[i]/con_set[i]
        conc_decode_basis = unit_decode_basis*brain_betabo_conc_table_df['conc'][i]
        real_add_conc_decode_basis += conc_decode_basis.real
        imag_add_conc_decode_basis += conc_decode_basis.imag
add_conc_decode_basis = real_add_conc_decode_basis + 1j*imag_add_conc_decode_basis

plt.figure(figsize=(10,5))
plt.title('add_conc_decode_basis')
plt.plot(crop_ppm,add_conc_decode_basis)
plt.xlim((4.5,0.5))

#################Boarden linewith###############
plt.figure(figsize=(10,5))
plt.title('Origin s data')
plt.plot(add_conc_decode_basis)

filted_tdata_ori = np.fft.ifft(add_conc_decode_basis)
plt.figure(figsize=(10,5))
plt.title('Origin t data')
plt.plot(filted_tdata_ori, color='b')

x_t = np.arange(0,len(filted_tdata_ori))

#t2 = 50#30,100,600 50 - 300

for t2 in range(20,70,5):   
    exp_adop_filt = np.exp(-(x_t/t2))
    filted_tdata = filted_tdata_ori*exp_adop_filt
    filted_sdata = np.fft.fft(filted_tdata)
    #plt.figure(figsize=(10,5))
    plt.figure(figsize=(10,5))
    plt.title(f'Apply gaussian, t2 ={t2}')
    plt.plot(filted_sdata)


##################MM Baseline###################
def g(x, A, μ, σ):
    return (A / (σ * math.sqrt(2 * math.pi))) * np.exp(-(x-μ)**2 / (2*σ**2))

group_name = ['MM09','MM12','MM14','MM16','MM20','MM21','MM23','MM26','MM30','MM31','MM37','MM38','MM40']
AU = ['0.72','0.28','0.38','0.05','0.45','0.36','0.36','0.04','0.2','0.11','0.64','0.07','1']
freq = ['0.9','1.21','1.38','1.63','2.01','2.09','2.25','2.61','2.96','3.11','3.67','3.8','3.96']
FWHM=['21.2','19.16','15.9','7.5','29.03','20.53','17.89','5.3','14.02','17.89','33.52','11.85','37.48']#Hz
MM_table = {
        "name": group_name,
        "AU":AU,
        "freq":freq,
        "FWHM":FWHM
        }
MM_table_df = pd.DataFrame(MM_table)

plt.figure(figsize=(10,5))
add_MM = np.zeros(crop_ppm.size).astype(float)
for i in range(len(MM_table_df)):
    amp = float(MM_table_df['AU'][i])*2500000
    fre_ppm = float(MM_table_df['freq'][i])
    w = (float(MM_table_df['FWHM'][i])/B)/2.355#FWHM to sigma
    mmm = g(crop_ppm,amp,fre_ppm,w)
    #print("mmm",mmm.real)
    plt.title(str(MM_table_df.loc[i]))
    plt.plot(crop_ppm,mmm.real)
    plt.xlim((4.5,0.5))
    add_MM += mmm.real
    '''
    if(i>10):
        add_MM += mmm.real
    if(i==12):
        plt.plot(crop_ppm,add_MM)
        plt.xlim((4.5,0.5))
    '''

plt.figure(figsize=(10,5))
plt.title('MM sum')
plt.plot(crop_ppm,add_MM)
plt.xlim((4.5,0.5))
plt.ylim((0,0.4*10**8))


plt.figure(figsize=(10,5))
plt.title('MM + conc')
plt.plot(crop_ppm,add_conc_decode_basis+add_MM)
plt.xlim((4.5,0.5))

##############################In vivo spectrum############################
from scipy.io import loadmat
a = loadmat(r'D:\1072MRI\HW2\meas_svs.mat')
tdata = a['tdata']
sdata = a['sdata']

sample_point = 2048
BW = 2000#Hz
B = 127#3T = 3*42.58= 127MHz
ppm_length = BW/B#16.xx ppm
ppm_center = 4.7

min_bound = ppm_center - (ppm_length)/2
max_bound = ppm_center + (ppm_length)/2
ppm = np.linspace(min_bound, max_bound, sample_point)
ppm = ppm[::-1]#reverse

vivo_x_range = np.where((ppm>0.5) & (ppm<4.5))  
vivo_x_range = np.array(vivo_x_range).reshape(-1)
ppm_vivo_x_range = ppm[vivo_x_range]
vivo_y_range = sdata[vivo_x_range,0]

plt.figure(figsize=(10,5))
plt.title('Vivo data')
plt.plot(ppm_vivo_x_range,vivo_y_range, color='r')
plt.xlim((4.5,0.5))

#直接是偏移量

'''
做papar p.16業的
將basis 乘以對應的conc濃度 組成一個像正常人腦的spectrum
然後參考paper上的配方模擬MM
'''