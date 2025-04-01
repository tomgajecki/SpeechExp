function p = call_map(varargin)
    p = ACE_map();
    p.num_selected = 8;
    p.base_level = 0.0156;
    p.sat_level = 0.5859;
    p.implant_stim_rate_Hz = 8000;
    p.processes([2, 3]) = [];
end