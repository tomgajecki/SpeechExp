function x = mix_audio(s, n, snr, fs)
% MIX_AUDIO Mixes speech and noise at a specified SNR.
%
%   x = mix_audio(s, n, snr, fs) returns the mixture x, where:
%       - s is the speech signal (a column vector).
%       - n is the noise signal (a column vector).
%       - snr is the desired signal-to-noise ratio in dB.
%       - fs is the sampling rate (in Hz). If not provided, it defaults to 16 kHz.
%
%   The function performs the following steps:
%       1. Removes the first 2 seconds from the speech signal.
%       2. Pads the remaining speech with 1 second of silence at the beginning
%          and at the end.
%       3. Selects a noise segment of the same length (randomly, or repeats noise
%          if it is too short).
%       4. Equalizes RMS values and scales the noise so that the SNR (using the
%          speech RMS as reference) matches the specified value.
%
%   Example usage:
%       [s, fs] = audioread('speech_file.wav');
%       [n, ~] = audioread('noise_file.wav');
%       x = mix_audio(s, n, 10, fs);
%
%   Author: Your Name
%   Date: YYYY-MM-DD

% Set default sampling rate if not provided
if nargin < 4
    fs = 16000;  % default to 16 kHz
end

% Ensure input signals are column vectors
s = s(:);
n = n(:);

% Step 1: Remove first 2 seconds from speech
numSamplesRemove = 2 * fs;
if length(s) <= numSamplesRemove
    error('Speech signal must be longer than 2 seconds.');
end
s_proc = s(numSamplesRemove+1:end);

% Step 2: Pad the processed speech with 1 second of silence at beginning and end
pad = zeros(fs, 1);
s_proc = [pad; s_proc];

% Desired length for the noise segment
L = length(s_proc);

% Step 3: Select (or repeat) noise segment of the same length
if length(n) >= L
    % Randomly choose a segment of length L
    start_idx = randi(length(n) - L + 1);
    noise_segment = n(start_idx:start_idx+L-1);
else
    % If noise is shorter than L, repeat it until reaching the desired length
    repFactor = ceil(L / length(n));
    noise_repeated = repmat(n, repFactor, 1);
    noise_segment = noise_repeated(1:L);
end

% Step 4: Equalize RMS and scale noise for desired SNR
rms_s = sqrt(mean(s_proc.^2));
rms_n = sqrt(mean(noise_segment.^2));

% Compute target noise RMS given the desired SNR (dB):
%   SNR (dB) = 20*log10(rms_s / rms_noise)
% So, target_noise_rms = rms_s / (10^(snr/20))
target_noise_rms = rms_s / (10^(snr/20));

% Scale factor to apply to the noise segment
scale = target_noise_rms / rms_n;
noise_scaled = noise_segment * scale;

% Mix speech and scaled noise
x = s_proc + noise_scaled;
x = x.';
end
