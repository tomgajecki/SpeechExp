function [r, data] = Get_MAP_details(r, map_number)

% Get_MAP_details: Retrieves recipient's MAP details from Cochlear database.
% Basic MAP details are retreived from the Cochlear_Database and converted
% into a parameter struct conforming with the NMT conventions.
% NOTE: Combinations of last- and first names in the database must be unique.
% Supported Database Versions: Custom Sound Suite 1.2.x and lower.
%
% r = Get_MAP_details(r, map_number)
%
% Inputs:
% r:                    Parameter struct with a fields:
% r.lastname:           Recipient's last name
% r.firstname:          Recipient's first name
% map_number:           Valid MAP number for this recipient
%
% Outputs:
% r:                    Parameter struct with the additional field:
% r.map(index):         Appended MAP parameter struct. "index" is the 
%                       incremented of the already existing map structs
%                       for this recipient
%
% See also: Gen_recipient, Get_data.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Copyright: Cochlear Ltd
%      $Change: 86418 $
%    $Revision: #1 $
%    $DateTime: 2008/03/04 14:27:13 $
%      Authors: Herbert Mauch
%               credits to Michael Büchler (USZ Zürich)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Create SQL query and retreive data
querytext = sprintf(['SELECT     Channel.ChannelNumber,'...
    'Channel.Electrode_Active, Channel.I_Enabled, Channel.TLevel_uCL,'...
    'Channel.CLevel_uCL, Channel.LowerFrequency_Hz, '...
    'Channel.UpperFrequency_Hz, Channel.RecordVersion, '...
    'Channel.AdditionalParameters, MAP.T_Strategy, '...
    'MAP.InitialPulseWidth_ns, MAP.InterPhaseGap_ns, MAP.Q,'...
    'MAP.T_ChannelStimOrder, MAP.Maxima, MAP.T_StimulationMode, '...
    'MAP.StimulationRate_Hz, MAP.TotalStimulationRate_Hz, MAP.BaseLevel, '...
    'MAP.SPLT_db, MAP.SPLC_db, MAP.MAPTitle, Implant.FK_Part FROM   '...
    'dbo.Recipient INNER JOIN dbo.Implant ON dbo.Recipient.GUID_Recipient = dbo.Implant.FK_GUID_Recipient INNER JOIN dbo.MAP ON dbo.Implant.GUID_Implant = dbo.MAP.FK_GUID_Implant CROSS JOIN  dbo.Channel WHERE     (Recipient.Name_Last = ''' r.lastname ''') AND (Recipient.Name_First = ''' r.firstname ''') AND (MAP.MAPNumber = ''' num2str(map_number) ''') AND (MAP.GUID_MAP = Channel.FK_GUID_MAP) AND (Implant.GUID_Implant = MAP.FK_GUID_Implant) AND (MAP.T_RecordStatus = 0) ORDER BY Channel.ChannelNumber']);       

        
data = Get_data(querytext);

%% convert data into NMT map data struct
% identify the active channels
active_channels = flipud(find(cell2mat(data.I_Enabled)));

% decode strategy
switch data.T_Strategy(1)
    case {1000000340,1000000035}
        strategy = 'ACE';
    case 1000000036
        strategy = 'SPEAK';
    case 1000000038
        strategy = 'CIS';
    otherwise
        strategy = 'unknown strategy';
end

% decode implant type
switch data.FK_Part(1)
    case 50001
        cic = 'CIC1';
    case {3712,5355,5891,7622,50002,50004}
        cic = 'CIC3';
    case {6800,6805,7623}
        cic = 'CIC4';
    otherwise
        cic = 'unknown implant type';
end

% decode stimulation mode
switch data.T_StimulationMode(1)
    case 1000000041
        mode = 103;
    case 1000000042
        mode = 101;
    case 1000000043
        mode = 102;
    otherwise
        mode = 'unknown mode';
end

% decode stimulation order
switch data.T_ChannelStimOrder(1)
    case {0, 1000000178}                % '0' is the default for ACE and SPEAK MAPS 
        order = 'base-to-apex';
    case 1000000179
        order = 'apex-to-base';
    otherwise
        order = 'unknown order';
end

if ~data.SPLT_db(1)                     % Legacy MAP with Base Level
    base_level = data.BaseLevel(1)/256;
else                                    % Freedom MAP with T-SPL and C-SPL
    base_level = 150/(256*From_dB(data.SPLC_db(1) - data.SPLT_db(1)));
end

% decode channel-electrode allocation for different DB record versions
switch data.RecordVersion(1)
    case 1  % CS v1.2 and lower record
        electrodes = data.Electrode_Active(active_channels);
    case 2  % CS v1.3 and higher record where channel electrode allocation is stored
            % in binary blob in AdditionalParameters field
        for e=1:length(active_channels)
            % convert binary into string
            text = char(cell2mat(data.AdditionalParameters(active_channels(e))))';
            i = 1;
            while i < length(text)      % scan for text header
                if strcmp(text(i:i+18), 'T_Electrode_Active=')
                    electrode = str2num(text(i+26:i+28)); 
                    if electrode > 222  % unsupported values
                        electrodes(e,1) = 0;
                    else
                        electrodes(e,1) = electrode - 200;
                    end
                    i = length(text);
                else
                    i = i + 1;
                end
            end
        end
    otherwise
        electrodes = zeros(22,1);
end
        
% read the data into a NMT struct
struct.map_number         = map_number;
struct.map_title          = cell2mat(data.MAPTitle(1));
struct.map_name           = strategy;
struct.channel_stim_rate  = data.StimulationRate_Hz(1);
struct.num_bands          = length(active_channels);
struct.num_selected       = data.Maxima(1);
struct.implant_stim_rate  = data.TotalStimulationRate_Hz(1);
struct.implant.IC         = cic;
struct.phase_width        = data.InitialPulseWidth_ns(1)/1000;
struct.phase_gap          = data.InterPhaseGap_ns(1)/1000;
struct.crossover_freqs    = [data.LowerFrequency_Hz(active_channels); data.UpperFrequency_Hz(min(active_channels))];
struct.Q                  = data.Q(1);
struct.base_level         = base_level;
struct.channel_order_type = order;
struct.electrodes         = electrodes;
struct.modes              = mode;
struct.threshold_levels   = round(data.TLevel_uCL(active_channels)/1000000);
struct.comfort_levels     = round(data.CLevel_uCL(active_channels)/1000000);

% add the map to the maps in the recipient struct
if isfield(r, 'map')
    i = length(r.map) + 1;
else
    i = 1;
end

r.map(i) = struct;