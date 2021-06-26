function wav = iFourier(Fwav)     
    wav = ifftshift(ifft2(ifftshift(Fwav)));
end