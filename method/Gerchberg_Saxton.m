function est_wav = Gerchberg_Saxton(GT_wav, image_num, OM)
    iter_num    = 100;

    est_wav = ones(OM.N);
    GT_Fwav = Fourier(OM.pad(GT_wav));    
    GT_Iimg = abs(GT_wav).^2;
    GT_Fimg = abs(GT_Fwav).^2;

    %% Capture images
    if OM.contain_noise
        Iimg_all = cell(1,image_num/2); 
        Fimg_all = cell(1,image_num/2); 
        
        for avg_iter = 1:image_num/2 
            [Iimg_all{avg_iter},SNR] = OM.addnoise(GT_Iimg);
            fprintf('SNR = %.3f\n', SNR ); 
            [Fimg_all{avg_iter},SNR] = OM.addnoise(GT_Fimg);
            fprintf('SNR = %.3f\n', SNR );                                       
        end
        Iimg = mean(cat(3,Iimg_all{:}),3);
        Fimg = mean(cat(3,Fimg_all{:}),3);      
    else
        Iimg = GT_Iimg;
        Fimg = GT_Fimg;
    end  
    
    %% Optimization    
    for i = 1:iter_num
        est_Fwav    = Fourier(OM.pad(est_wav));
        est_Fwav    = sqrt(Fimg).*exp(1j*angle(est_Fwav));
        est_wav     = OM.crop(iFourier(est_Fwav));
        est_wav     = sqrt(Iimg).*exp(1j*angle(est_wav));  
        figure(1);
        err = OM.evaluate_result(GT_wav,est_wav);
        fprintf('iter %d, err = %f\n' ,i, err);
    end       
end