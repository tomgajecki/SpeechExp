function p = call_map(name, surname, side, mapNumber)
    d = Gen_recipient(surname, name);
    p = Get_MAP_details(d, mapNumber).map;
    p = ACE_map(p);
    p.upper_levels = p.comfort_levels;
    p.lower_levels = p.threshold_levels;
    p.processes([2, 3]) = [];
end