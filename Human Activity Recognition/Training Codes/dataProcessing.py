import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from IPython.display import display
plt.style.use('bmh')
Raw_data_paths = 'serhatDataset/Processing/'
total_accel = pd.read_csv(Raw_data_paths + 'total_acceleration.csv')
total_gyro = pd.read_csv(Raw_data_paths + 'gyro.csv')
labels = pd.read_csv(Raw_data_paths + 'labels.txt',header=None, delim_whitespace=True)
print(total_accel.head())
print(total_gyro.head())
sampling_freq=50
def visualize_signal(signal,x_labels,y_labels,title,legend):
    plt.figure(figsize=(15,8))
    time=[1/float(sampling_freq) *i for i in range(len(signal))] # convert row numbers in time durations
    # plotting the signal
    plt.plot(time,signal,label=legend) # plot the signal and add the legend
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show() # show the figure
from scipy.signal import medfilt # import the median filter function

def median(signal):# input: numpy array 1D (one column)
    array=np.array(signal)   
    #applying the median filter
    med_filtered=sp.signal.medfilt(array, kernel_size=3) # 3rd order median filter
    return  med_filtered # return the med-filtered signal: numpy array 1D
legend1='sample accelaration X-axis'
legend2='median filterd sample accelaration X-axis'
x_labels='time in seconds'
y_labels='acceleration amplitude in 1g'
title1='8 seconds of the original signal'
title2='the same 8 seconds after applying the median filter order 3'
signal_sample=np.array(total_accel['acc_X'])
med_filtred_signal=median(signal_sample)
#visualize_signal(signal_sample[300:600],x_labels,y_labels,title1,legend1) 
#visualize_signal(med_filtred_signal[300:600],x_labels,y_labels,title2,legend2)
# signal processing part 
nyqRate = sampling_freq/float(2)
freq1 = 0.3 # filter cutoff freq, to remove DC componant as specified in the data set
freq2 = 20 # filter cutoff frq, for body componant
from scipy.fftpack import fft 
def filtering_and_extraction(signalT, freq1, freq2):
    signalT = np.array(signalT)
    signalT_length = len(signalT)
    signalF = fft(signalT)
    allFreqs = np.array(sp.fftpack.fftfreq(signalT_length, d=1/float(sampling_freq))) # return all freq in range of (-25Hz, 25Hz)
    bodySignalF = []
    noiseF = []
    for i in range(len(allFreqs)):
        if(abs(allFreqs[i] <= freq1 or abs(allFreqs[i]) > freq2)):
            bodySignalF.append(float(0))
        else:
            bodySignalF.append(signalF[i])
        if(abs(allFreqs[i] <= freq2)):
            noiseF.append(float(0))
        else:
            noiseF.append(signalF[i])
    bodySignalT = sp.fftpack.ifft(np.array(bodySignalF)).real
    noiseT =  sp.fftpack.ifft(np.array(noiseF)).real
    totalAcceleration = signalT - noiseT
    return (totalAcceleration, bodySignalT, noiseT)
filtred_signal,_,_ = filtering_and_extraction(signal_sample, freq1, freq2)
legend2='filterd sample accelaration X-axis'
#visualize_signal(signal_sample[300:600],x_labels,y_labels,title1,legend1) 
#visualize_signal(filtred_signal[300:600],x_labels,y_labels,title2,legend2)
# apply the 3rd order median and  filtering to raw data gyro and accel
timeSignalAll = pd.DataFrame()
for column in total_accel.columns:
    eachAxisSignal = np.array(total_accel[column])
    medianFilteredSignal = median(eachAxisSignal) # apply 3rd order median filter
    totalAcc,bodyAcc,_ = filtering_and_extraction(medianFilteredSignal,freq1,freq2) # seperate the body componant
    timeSignalAll['t_body_'+column] = bodyAcc[:-1]
    timeSignalAll['total_' + column] = totalAcc[:-1]
for column in total_gyro.columns:
    eachAxisSignalGyro = np.array(total_gyro[column])
    medianFilteredSignalGyro = median(eachAxisSignalGyro)
    _,bodyGyro,_ = filtering_and_extraction(medianFilteredSignalGyro,freq1,freq2)
    timeSignalAll['t_body_' + column] = bodyGyro[:-1]
newOrderColumn = ['t_body_acc_X', 't_body_acc_Y', 't_body_acc_Z', 't_body_gyro_X', 't_body_gyro_Y', 't_body_gyro_Z',
                 'total_acc_X', 'total_acc_Y', 'total_acc_Z']
timeSignalAllOrdered = pd.DataFrame()
for column in newOrderColumn:
    timeSignalAllOrdered[column] = timeSignalAll[column]
display(timeSignalAllOrdered.shape) # the of the first dataframe
display(timeSignalAllOrdered.describe()) # dataframe's statistics
timeSignalAllOrdered.head(3) # displaying the fisrt three rows

# Windowing 

def normalize5(number): 
    stre=str(number)
    if len(stre)<5:
        l=len(stre)
        for i in range(0,5-l):
            stre="0"+stre
    return stre 
def normalize2(number):
    stre=str(number)
    if len(stre)<2:
        stre="0"+stre
    return stre
windowedLabels = []
def windowing (timeSignals, labels):
    columns = timeSignals.columns
    windowedSignals = {}
    windowID = 0
    labelArray = np.array(labels[labels[0] < 8])
    for line in labelArray:
        activityID = line[0]
        starting = line[1]
        ending = line[2]
        for cursor in range(starting, ending - 127, 64):
            endPoint = cursor + 128
            data = np.array(timeSignals.iloc[cursor:endPoint])
            window = pd.DataFrame(data = data, columns = columns)
            key = 't_W'+normalize5(windowID)+'_act'+normalize2(activityID)
            windowedSignals[key] = window
            windowID = windowID + 1
            windowedLabels.append(activityID)
    return windowedSignals
t_windowedSignal = windowing(timeSignalAllOrdered, labels)
print(t_windowedSignal['t_W00081_act04'].head(128))
print(t_windowedSignal['t_W00081_act04'].shape)
#print(list(t_windowedSignal.keys()))
print(len(t_windowedSignal))
print(len(windowedLabels))
windLabel_dataframe = pd.DataFrame(windowedLabels, columns=['Labels'])
path1="New Data\\After processing\\labels.csv"
windLabel_dataframe.to_csv(path_or_buf=path1, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
# output files
body_acc_x = pd.DataFrame()
body_acc_y = pd.DataFrame()
body_acc_z = pd.DataFrame()
body_gyro_x = pd.DataFrame()
body_gyro_y = pd.DataFrame()
body_gyro_z = pd.DataFrame()
total_acc_x = pd.DataFrame()
total_acc_y = pd.DataFrame()
total_acc_z = pd.DataFrame()
for key in t_windowedSignal:
    sample = t_windowedSignal[key].transpose()
    body_acc_x = body_acc_x.append(sample.loc['t_body_acc_X'], ignore_index=True)
    body_acc_y = body_acc_y.append(sample.loc['t_body_acc_Y'], ignore_index=True)
    body_acc_z = body_acc_z.append(sample.loc['t_body_acc_Z'], ignore_index=True)
    body_gyro_x = body_gyro_x.append(sample.loc['t_body_gyro_X'], ignore_index=True)
    body_gyro_y = body_gyro_y.append(sample.loc['t_body_gyro_Y'], ignore_index=True)
    body_gyro_z = body_gyro_z.append(sample.loc['t_body_gyro_Z'], ignore_index=True)
    total_acc_x = total_acc_x.append(sample.loc['total_acc_X'], ignore_index=True)
    total_acc_y = total_acc_y.append(sample.loc['total_acc_Y'], ignore_index=True)
    total_acc_z = total_acc_z.append(sample.loc['total_acc_Z'], ignore_index=True)
print(body_acc_x.head())
print(body_acc_x.shape)
# export to csv files
windLabel_dataframe.to_csv(path_or_buf=path1, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path2="New Data\\After processing\\body_acc_x.csv"
body_acc_x.to_csv(path_or_buf=path2, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path3="New Data\\After processing\\body_acc_y.csv"
body_acc_y.to_csv(path_or_buf=path3, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path4="New Data\\After processing\\body_acc_z.csv"
body_acc_z.to_csv(path_or_buf=path4, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path5="New Data\\After processing\\body_gyro_x.csv"
body_gyro_x.to_csv(path_or_buf=path5, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path6="New Data\\After processing\\body_gyro_y.csv"
body_gyro_y.to_csv(path_or_buf=path6, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path7="New Data\\After processing\\body_gyro_z.csv"
body_gyro_z.to_csv(path_or_buf=path7, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path8="New Data\\After processing\\total_acc_x.csv"
total_acc_x.to_csv(path_or_buf=path8, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path9="New Data\\After processing\\total_acc_y.csv"
total_acc_y.to_csv(path_or_buf=path9, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
path10="New Data\\After processing\\total_acc_z.csv"
total_acc_z.to_csv(path_or_buf=path10, na_rep='NaN', columns=None, header=False, index=False, mode='w', encoding='utf-8',line_terminator='\n',)
