function [err,phase_cor] = wav_loss(GT_wav, est_wav)
    phi_all = (0:11)/12*2*pi;    
    rms_loss     = @(x) sqrt(mean(mean(abs(x).^2))); 
    err_all = arrayfun(@(phi) rms_loss(GT_wav-exp(1j*phi)*est_wav), phi_all);
    [~,mid] = min(err_all);
    est_wav = exp(1j*phi_all(mid))*est_wav;
    diff_wav = est_wav.*conj(GT_wav);
    phi_err = angle(mean(diff_wav(:)));
    est_wav = est_wav* exp(-1j* phi_err);
    phase_cor = phi_all(mid) - phi_err;
    err = rms_loss(GT_wav-est_wav)/rms_loss(GT_wav);
end