%% Detect Commands Using Streaming Audio from Microphone
% Test your newly trained command detection network on streaming audio from
% your microphone. If you have not trained a network, then type
% |load('commandNet.mat')| at the command line to load a pretrained network
% and the parameters required to classify live, streaming audio. Try
% speaking one of the speech commands, for example, 'yes', 'no', or 'stop'.
% Then, try one of the unknown words, such as 'marvin', 'sheila', 'bed',
% 'house', 'cat', 'bird', or any number from zero to nine.


%%
% Specify the audio sampling rate and classification rate in Hz and create
% an audio device reader to read audio from your microphone.
fs = 16e3;
classificationRate = 20;
audioIn = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',floor(fs/classificationRate));

%%
% Specify parameters for the streaming spectrogram computations and
% initialize a buffer for the audio. Extract the classification labels of
% the network and initialize buffers of half a second for the labels and
% classification probabilities of the streaming audio. Use these buffers to
% build 'agreement' over when a command is detected using multiple frames
% over half a second.
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;
epsil = 1e-6;
specMin = min(XTrain(:));
specMax = max(XTrain(:));

frameLength = frameDuration*fs;
hopLength = hopDuration*fs;
waveBuffer = zeros([fs,1]);

YBuffer(1:classificationRate/2) = "unknown";
probBuffer = zeros([numel(labels),classificationRate/2]);

%%
% Create a figure and detect commands as long as the created figure exists.
% To stop the live detection, simply close the figure. Add the path of the
% <matlab:edit(fullfile(matlabroot,'examples','audio','main','auditorySpectrogram.m'))
% |auditorySpectrogram|> function that calculates the spectrograms.
h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
addpath(fullfile(matlabroot,'examples','audio','main'))

while ishandle(h)
    
    % Extract audio samples from audio device and add to the buffer.
    x = audioIn();
    waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
    waveBuffer(end-numel(x)+1:end) = x;
   % waveBuffer = zscore(waveBuffer, [],1);
    % Compute the spectrogram of the latest audio samples.
    spec = auditorySpectrogram(waveBuffer,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength-hopLength, ...
        'NumBands',numBands, ...
        'Range',[50,7000], ...
        'WindowType','Hann', ...
        'WarpType','Bark', ...
        'SumExponent',2);
    spec = log10(spec + epsil);
    spec_not_norm = spec;
    spec_not_norm_disp = spec;
    
    
    
    
    spec_not_norm = reshape(spec_not_norm, 1, size(spec_not_norm, 1)*size(spec_not_norm, 2));
    spec = spec_not_norm; %(spec_not_norm - repmat(mean_Train, size(spec_not_norm, 1))) ./ repmat(std_Train, size(spec_not_norm, 1), 1);
    % Classify the current spectrogram, save the label to the label buffer,
    % and save the predicted probabilities to the probability buffer.   
    [prediction] = predict(Theta1, Theta2, spec);
    h1 = sigmoid([ones(1, 1) spec] * Theta1');
    h2 = sigmoid([ones(1, 1) h1] * Theta2');
    prob = h2;
    YBuffer(1:end-1)= YBuffer(2:end);
    YBuffer(end) = labels(prediction);
    probBuffer(:,1:end-1) = probBuffer(:,2:end);
    probBuffer(:,end) = prob;
    % Plot the current waveform and spectrogram.
    subplot(2,1,1);
    plot(waveBuffer)
    axis tight
    ylim([-0.2,0.2])
    
    subplot(2,1,2)
    pcolor(spec_not_norm_disp)
    caxis([specMin+2 specMax])
    shading flat
    
    % Now do the actual command detection by performing a very simple
    % thresholding operation. Declare a detection and display it in the
    % figure title if all of the following hold: 
    % 1) The most common label is not |background|. 
    % 2) At least |countThreshold| of the latest frame labels agree. 
    % 3) The maximum predicted probability of the predicted label is at least |probThreshold|. 
    % Otherwise, do not declare a detection.
    YBuffer = categorical(YBuffer);
    [YMode,count] = mode(YBuffer);
    countThreshold = ceil(classificationRate*0.2);
    maxProb = max(probBuffer(labels == string(YMode),:));
    probThreshold = 0.6;
    subplot(2,1,1);
    if YMode == "unknown" || count<countThreshold || maxProb < probThreshold
        title(" ")
    else
        title([string(YMode) num2str(maxProb)],'FontSize',20)
    end
    drawnow
end