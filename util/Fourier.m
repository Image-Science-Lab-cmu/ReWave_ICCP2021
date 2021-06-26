function Fwav = Fourier(wav)     
    Fwav = fftshift(fft2(fftshift(wav)));
end