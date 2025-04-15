function stream(p, x)
    % STREAM Process and stream audio or LGF data through the NIC
    %   STREAM(p, x) processes the input data x through the NIC with parameters p
    %   and streams the result to the audio output.
    %
    %   Inputs:
    %       p - Parameters structure for the NIC
    %       x - Input data (either audio waveform or LGF data)
    %
    %   The function automatically detects whether x is audio data or LGF data
    %   and processes it accordingly.

    % Check if x is audio data or LGF data
    % Audio data is typically a 1D vector, while LGF data is a 2D matrix
    if isvector(x) || (ismatrix(x) && size(x, 1) == 1)
        % x is audio data
        q = Process(p, x);
    else
        % x is LGF data
        [~, frames] = size(x);
        originalTime = linspace(0, 1, frames);  % Normalized time from 0 to 1
        newTime = linspace(0, 1, p.channel_stim_rate*frames/1000);
        x = interp1(originalTime, x', newTime, 'makima')';  
        x_filtered = x(1:p.num_bands, :);
        q = Process_from_lgf(p, x_filtered);
    end
    
    % Get NIC properties
    %jp = NIC_properties(p);
    
    % Create and start NIC streamer
    %s = NIC_streamer(jp);
    %s.start();
    %s.stream(q);
    %s.wait();
    %s.stop();
    
    % Optional: Plot the sequence
    % Plot_sequence(q);
    
    % Process through RF Vocoder and play the result
    v = RFVocoder(p);
    s = Process(v, q);
    soundsc(s, p.audio_sample_rate_Hz);
end
