# MATLAB Functions

This directory contains the required MATLAB functions for the Speech Intelligibility Experiment application.

## Required Functions

### `manageStreamServer.m`
```matlab
function manageStreamServer(action)
    % Manages the stream server for audio playback
    % action: 'start' or 'stop'
    
    persistent streamServer
    
    switch action
        case 'start'
            if isempty(streamServer)
                streamServer = audioplayer([], 44100);
            end
        case 'stop'
            if ~isempty(streamServer)
                stop(streamServer);
                streamServer = [];
            end
    end
end
```

### `ACE_map.m`
```matlab
function map = ACE_map()
    % Creates a default ACE map for cochlear implant stimulation
    % Returns a map structure with default parameters
    
    map = struct();
    map.sampleRate = 44100;
    map.numChannels = 22;
    map.pulseWidth = 25e-6;  % 25 microseconds
    map.phaseGap = 8e-6;     % 8 microseconds
    map.stimRate = 500;      % 500 Hz per channel
    map.maxLevel = 255;      % Maximum stimulation level
    map.minLevel = 0;        % Minimum stimulation level
end
```

### `stream.m`
```matlab
function stream(map, audioFile)
    % Streams an audio file through the cochlear implant map
    % map: ACE map structure
    % audioFile: path to the WAV file to stream
    
    % Read the audio file
    [audio, fs] = audioread(audioFile);
    
    % Resample if necessary
    if fs ~= map.sampleRate
        audio = resample(audio, map.sampleRate, fs);
    end
    
    % Process audio through the map
    % (Implementation details depend on your specific CI processing algorithm)
    
    % Play the processed audio
    player = audioplayer(audio, map.sampleRate);
    play(player);
    
    % Wait for playback to complete
    while isplaying(player)
        pause(0.1);
    end
end
```

## Usage

These MATLAB functions should be placed in your MATLAB path. The application will call them through the MATLAB Engine API.

### Function Dependencies
- `manageStreamServer.m`: Manages the audio stream server
- `ACE_map.m`: Creates default CI map parameters
- `stream.m`: Processes and plays audio through the CI map

### Notes
- The `stream.m` function is a placeholder - you'll need to implement the actual CI processing algorithm
- Adjust the parameters in `ACE_map.m` according to your specific CI requirements
- The sample rate (44100 Hz) can be modified if needed 