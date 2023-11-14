fs = 48000
t = 18
times = 0:(1/fs):(t-1/fs);
mask = pinknoise(t*fs);

plot(times,mask')

%%
nSamp = length(mask)
fadeIN_dur = 1*fs;
fadeOUT_dur = 1*fs;
% Create a fading envelope
fadeSignal = [linspace(0, 1,fadeIN_dur), ones(1,nSamp-fadeOUT_dur-fadeIN_dur),linspace(1, 0, fadeOUT_dur)]';
% If stereo, make another column
if size(mask, 2) > 1
	fadeSignal = [fadeSignal, fadeSignal];
end

% Plot fading signal.
subplot(3, 1, 2);
plot(times,fadeSignal);
%title('Fading Envelope Signal', 'FontSize', fontSize);
grid on;

%%
mask_fadein = fadeSignal.* mask;
%%
%sound(mask_fadein,fs);
audiowrite("PinkNoise18s.wav",mask_fadein,fs);
