clc
clear

modulationTypes = categorical(sort(["BPSK", "QPSK", "PSK_8", ...
    "QAM_16", "QAM_64", "PAM_4", "GFSK", "CPFSK"]));

% define parameters
numFramesPerModType = 1000;

sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
fs = 200e3;             % Sample rate
fc = 100e6;             % Center frequencies
SNR_range = -20:2:18;   % SNR range from -20 to 18 dB in steps of 2
WLEN = 128 ;            % Output Frame length 

channel_1 = helperModClassTestChannel(...
    'SampleRate', fs, ...
    'PathDelays', [0 0.9 1.7] / fs, ...
    'AveragePathGains', [0 -1 -5], ...
    'KFactor', 4, ...
    'MaximumDopplerShift', 1, ...
    'MaximumClockOffset', 0.5, ...
    'CenterFrequency', fc);


channel_2 = helperModClassTestChannel(...
    'SampleRate', fs, ...
    'PathDelays', [0 0.9 1.7]/ fs, ...
    'AveragePathGains', [0 -1 -5], ...
    'MaximumDopplerShift', 1, ...
    'MaximumClockOffset', 0.5, ...
    'CenterFrequency', fc, ...
    'UseRayleighChannel', true);

rng(12)

numModulationTypes = length(modulationTypes);
% channelInfo = info(channel);
transDelay = 50;

% Initialize storage for examples
examples_dict = struct();

tic

% this for loop passes each frame through channel impairments
% and the samples out from the output and then saves it 

for modType = 1:numModulationTypes


    label = modulationTypes(modType);
    elapsedTime = seconds(toc);
    elapsedTime.Format = 'hh:mm:ss';
    fprintf('%s -- Generating %s mdoualtion\n', elapsedTime, label)

    numSymbols = (numFramesPerModType / sps);
    dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf);
    modulator = helperModClassGetModulator(modulationTypes(modType), sps);

    for idx = 1:length(SNR_range)

        SNR = SNR_range(idx);
        channel_1.SNR = SNR;
        channel_2.SNR = SNR;

        channel = channel_1;

        for p=1:numFramesPerModType

            % Generate random data
            x = dataSrc();

            % Modulate
            y = modulator(x);

            % Pass through independent channels
            rxSamples = channel(y);

            % Remove transients from the beginning, trim to size, and normalize
            frame = helperModClassFrameGenerator(rxSamples, WLEN, 64, transDelay, sps);

            % Package data
            frame = frame' ;
            output_size = size(frame);
            examples = zeros(output_size(1), 1, 2, WLEN);
            examples(:, 1, 1, :) = real(frame(:,:));
            examples(:, 1, 2, :) = imag(frame(:,:));
            
            % changing + and - signs into pos and neg so it can be saved
            if SNR < 0
                SNR_str = sprintf('neg%d', abs(SNR));
            else
                SNR_str = sprintf('pos%d', SNR);
            end

            % Generate the field name
            fieldName = sprintf('%s__%s', label, SNR_str);

            % Check if the field exists in the struct
            if isfield(examples_dict, fieldName)
                % Append data if the field exists
                existingData = examples_dict.(fieldName);
                examples_dict.(fieldName) = [existingData; examples]; % Append new data
            else
                % Create the field if it does not exist
                examples_dict.(fieldName) = examples;
            end

            if p == numFramesPerModType/2 % change channel to Reyleigh
                channel = channel_2;
            end
        end
    end
end

elapsedTime = seconds(toc);
elapsedTime.Format = 'hh:mm:ss';
fprintf('%s -- end of generation\n', elapsedTime)

Save the data
save('modulation_examples.mat', '-struct', 'examples_dict');