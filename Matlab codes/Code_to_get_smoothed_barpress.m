%% Bar pressings smoothed
clear
clc

bar_pressings_smooth = load('./Smoothed_Bar_Press_Data/Recall_animal_barpress_smoothed.mat');
obx_smooth_press = table2array(bar_pressings_smooth.SRSmooth(:,'OB9'));
timing_smooth = bar_pressings_smooth.SR_New_Time;


sampletime = timing_smooth(end)/length(timing_smooth);
nsamples1200 = floor(1200/sampletime);

sbp = obx_smooth_press(1:nsamples1200);
smoothtime = timing_smooth(1:nsamples1200);

avg_smooth=[];
x=2;

for i = 1:floor(x/sampletime):nsamples1200-floor(x/sampletime)+1
    disp(i)
   	chunk = mean(sbp(i:i+floor(x/sampletime)-1));
    avg_smooth=[avg_smooth,chunk];
    
end
floor(x/sampletime)
%%
writematrix(avg_smooth,'avg_smooth.csv')
%%
%plot(timing_smooth(1:floor(2/sampletime):nsamples1200-floor(2/sampletime)+1),avg_smooth)
%hold on
%plot(timing_smooth,obx_smooth_press)
%%
length(avg_smooth)