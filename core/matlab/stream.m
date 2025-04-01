function stream(p, path)

    q = Process(p, path);
    jp = NIC_properties(p);
    s = NIC_streamer(jp);
    s.start();
    s.stream(q);
    s.wait();
    s.stop();
    %Plot_sequence(q);
    v = RFVocoder(p);
    s = Process(v, q);
    soundsc(s, p.audio_sample_rate_Hz);
end
