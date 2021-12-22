%% Analyze only some intervals
clear
clc
%addpath('C:\Users\Tomas\Github\ML_course_C\project2\Data\2020 Paired Stim Data Sets\PAC');
brain_stimuli = load('Data/OB9/raw_ephys_data_recall_OB9.mat');%SLOW
%%
BLA_amplitude_1 = brain_stimuli.BLA_data(:,1);
BLA_amplitude_2 = brain_stimuli.BLA_data(:,2);
BLA_amplitude_3 = brain_stimuli.BLA_data(:,3);
BLA_amplitude_4 = brain_stimuli.BLA_data(:,4);

IL_amplitude_1 = brain_stimuli.IL_data(:,1);
IL_amplitude_2 = brain_stimuli.IL_data(:,2);
IL_amplitude_3 = brain_stimuli.IL_data(:,3);
IL_amplitude_4 = brain_stimuli.IL_data(:,4);

time_origin = brain_stimuli.time_stamps_ephys;
fs = 30e3;

%%
m = 0;

for xs = 1:2*fs:1200*fs+1 %each 2 second one 1-second sample until 1200 seconds
    m=m+1;
    BLA_amplitude1 = BLA_amplitude_1(xs:xs+fs-1);
    BLA_amplitude2 = BLA_amplitude_2(xs:xs+fs-1);
    BLA_amplitude3 = BLA_amplitude_3(xs:xs+fs-1);
    BLA_amplitude4 = BLA_amplitude_4(xs:xs+fs-1);
    
    IL_amplitude1 = IL_amplitude_1(xs+1:xs+fs-1);
    IL_amplitude2 = IL_amplitude_2(xs+1:xs+fs-1);
    IL_amplitude3 = IL_amplitude_3(xs+1:xs+fs-1);
    IL_amplitude4 = IL_amplitude_4(xs+1:xs+fs-1);
    
    time = time_origin(xs:xs+fs-1);%1 sec length each sample
    disp('time, ampli calculated')


    BLA_power1 = fft(BLA_amplitude1); 
    BLA_power2 = fft(BLA_amplitude2); 
    BLA_power3 = fft(BLA_amplitude3); 
    BLA_power4 = fft(BLA_amplitude4); 
    n = length(BLA_power1);  
    f = (0:n-1)*fs/n;
    disp('power calculated')


%     % Time/frequency visualization
%     hFig = figure;
%     subplot(5,6,[1,2,3])
%     hold on
%     plot(time, BLA_amplitude1,'DisplayName','Time signal ch1')
%     plot(time, BLA_amplitude2,'DisplayName','Time signal ch2')
%     plot(time, BLA_amplitude3,'DisplayName','Time signal ch3')
%     plot(time, BLA_amplitude4,'DisplayName','Time signal ch4')
%     hold off
%     xlim([time(1),time(end)])
%     grid minor
%     xlabel('Time (s)');
%     ylabel('Amplitude (V)')
%     legend show
%     disp('time visualization')

%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[7,8,9])
%     hold on
%     plot(f,abs(BLA_power1)/n,'DisplayName','Spectrum amplitude ch1')
%     plot(f,abs(BLA_power2)/n,'DisplayName','Spectrum amplitude ch2')
%     plot(f,abs(BLA_power3)/n,'DisplayName','Spectrum amplitude ch3')
%     plot(f,abs(BLA_power4)/n,'DisplayName','Spectrum amplitude ch4')
%     hold off
%     xlabel('Frequency (Hz)')
%     ylabel('Norm. power (Wn)')
%     xlim([0,150])
%     grid minor
%     legend show 
%     disp('frequency visualization')
% 
% 
%     subplot(5,6,[13,14,15])
%     hold on
%     plot(f,angle(BLA_power1)*180/3.149,'DisplayName','Spectrum phase ch1')
%     plot(f,angle(BLA_power2)*180/3.149,'DisplayName','Spectrum phase ch2')
%     plot(f,angle(BLA_power3)*180/3.149,'DisplayName','Spectrum phase ch3')
%     plot(f,angle(BLA_power4)*180/3.149,'DisplayName','Spectrum phase ch4')
%     hold off
%     xlabel('Frequency (Hz)')
%     ylabel('Phase (º)')
%     xlim([0,150])
%     grid minor
%     legend show
%     disp('phase visualization')

    % Bandpass analysis
    theta1 = bandpass(BLA_amplitude1,[4,12],fs,'ImpulseResponse','iir','Steepness',1);
    theta2 = bandpass(BLA_amplitude2,[4,12],fs,'ImpulseResponse','iir','Steepness',1);
    theta3 = bandpass(BLA_amplitude3,[4,12],fs,'ImpulseResponse','iir','Steepness',1);
    theta4 = bandpass(BLA_amplitude4,[4,12],fs,'ImpulseResponse','iir','Steepness',1);
    
    gamma1 = bandpass(BLA_amplitude1,[50,100],fs,'ImpulseResponse','iir','Steepness',1);
    gamma2 = bandpass(BLA_amplitude2,[50,100],fs,'ImpulseResponse','iir','Steepness',1);
    gamma3 = bandpass(BLA_amplitude3,[50,100],fs,'ImpulseResponse','iir','Steepness',1);
    gamma4 = bandpass(BLA_amplitude4,[50,100],fs,'ImpulseResponse','iir','Steepness',1);
    
    
    theta = (theta1+theta2+theta3+theta4)/4;
    gamma = (gamma1+gamma2+gamma3+gamma4)/4;
    
    disp('bandpass analysis done')

%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[4,5,6])
%     plot(time,theta,'r','DisplayName','time signal theta avg: 4-12 Hz')
%     hold on
%     plot(time,gamma,'b','DisplayName','time signal gamma avg: 50-100 Hz')
%     xlabel('Time (s)')
%     ylabel('Amplitude (V)')
%     legend show
%     grid minor
%     hold off
%     disp('theta, gamma time')
    
%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[10,11,12])
%     plot((0:length(theta)-1)*fs/length(theta), abs(fft(theta))/length(theta),'r','DisplayName','spectrum theta: 4-12 Hz')  
%     hold on
%     plot((0:length(gamma)-1)*fs/length(gamma), abs(fft(gamma))/length(gamma),'b','DisplayName','spectrum gamma: 50-100 Hz')  
%     xlim([0,150])
%     xlabel('Frequency (Hz)')
%     ylabel('Norm. power (Wn)')
%     legend show
%     grid minor
%     hold off
%     disp('theta, gamma frequ')

%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[16,17,18])
%     plot((0:length(theta)-1)*fs/length(theta), angle(fft(theta))*180/3.149,'r','DisplayName','phase theta: 4-12 Hz')   
%     hold on
%     plot((0:length(gamma)-1)*fs/length(gamma), angle(fft(gamma))*180/3.149,'b','DisplayName','phase gamma: 50-100 Hz')    
%     xlabel('Frequency (Hz)')
%     xlim([0,150])
%     ylabel('Phase (º)')
%     legend show
%     grid minor
%     hold off
%     disp('theta, gamma phase')

    % Modulation Index (MI)
    % Envelope of HF to Phase of LF correlation ~ PAC (Phase-Amplitude Correlation)

     gamma_f = 40:1:150; 
     theta_f = 4:0.2:12; disp('frequ calculations')

    mi = find_pac_shf(gamma,fs,'mi',theta,theta_f,gamma_f,'n',1,7,200,50,0.05,'PAC MI analysis','Theta','Gamma'); disp('mi done')
%     
%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[19,20,25,26]) 
%     imagesc(theta_f,gamma_f, mi) 
%     colorbar 
%     title('PAC MI') 
%     xlabel('Theta freq (Hz) - Phase modulator') 
%     ylabel('G. freq (Hz) - A. modulated')
%     set(gca,'YDir','normal')

%     sgtitle('Interval BLA analysis')
    
    % Average analysis
    ampli_avg1 = bandpass(BLA_amplitude1,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    ampli_avg2 = bandpass(BLA_amplitude2,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    ampli_avg3 = bandpass(BLA_amplitude3,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    ampli_avg4 = bandpass(BLA_amplitude4,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    ampli_avg = (ampli_avg1+ampli_avg2+ampli_avg3+ampli_avg4)/4;
    power_avg = abs(fft(ampli_avg));
    n_avg = length(power_avg);  
    f_avg = (0:n_avg-1)*fs/n_avg;
%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[21,22,27,28])
%     plot(f_avg(1:100),power_avg(1:100)/n_avg)
%     title('Average BLA bandpower')
%     xlabel('Freq (Hz)')
%     ylabel('Norm. power (Wn)')

    
    
    % IL analysis
    % Average analysis
    IL_ampli_avg1 = bandpass(IL_amplitude1,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    IL_ampli_avg2 = bandpass(IL_amplitude2,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    IL_ampli_avg3 = bandpass(IL_amplitude3,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    IL_ampli_avg4 = bandpass(IL_amplitude4,[4,100],fs,'ImpulseResponse','iir','Steepness',1);
    IL_ampli_avg = (IL_ampli_avg1+IL_ampli_avg2+IL_ampli_avg3+IL_ampli_avg4)/4;
    IL_power_avg = abs(fft(IL_ampli_avg));
    IL_n_avg = length(IL_power_avg);  
    IL_f_avg = (0:IL_n_avg-1)*fs/IL_n_avg;
%     set(0,'CurrentFigure',hFig)
%     subplot(5,6,[23,24,29,30])
%     plot(IL_f_avg(1:100),IL_power_avg(1:100)/IL_n_avg)
%     title('Average IL bandpower')
%     xlabel('Freq (Hz)')
%     ylabel('Norm. power (Wn)')

    
% 
%     % Fill your structure
% 
%     s(m).theta_avg = theta;
%     s(m).gamma_avg = gamma;
%     s(m).time = time;
%     s(m).theta_power_avg = abs(fft(theta))/length(theta);
%     s(m).gamma_power_avg = abs(fft(gamma))/length(gamma);
%     s(m).f = f';
    s(m).mi = mi;
%     s(m).theta_f_pac = theta_f;
%     s(m).gamma_f_pac = gamma_f;
    s(m).bla_avg_power = power_avg(1:100)/n_avg;
    s(m).il_avg_power = IL_power_avg(1:100)/IL_n_avg;


    %saveas(gcf,[num2str(m),'_comp.fig'])
disp('sample')
disp(xs)
end
%%

structural_data = s;
%%
writetable(struct2table(structural_data), 'Data/struct_BLA_ch_avg600.csv')
