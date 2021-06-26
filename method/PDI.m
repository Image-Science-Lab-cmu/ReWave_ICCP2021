function [est_wav, err_all] = PDI(GT_wav, image_num, OM)
    %% Param initialization
    iter_num    = floor( (image_num-2)/2);         
    err_all     = zeros(1,iter_num);
    Mimgs       = zeros(OM.N,OM.N,3);

    %% Assume pinhole open at brightest position
    GT_Fwav     = Fourier(OM.pad(GT_wav));    
    Fimg        = abs(GT_Fwav).^2;        
    [~,id]      = max(Fimg(:));
    pinhole     = zeros(OM.PN); %assume pinhole has wave resolution
    pinhole(id) = 1;      
        
    for i = 1:iter_num    
        %% Interferometry between wave and pinhole        
        omega = (-1+sqrt(3)*1j)/2;
        masks = 1 + repmat(pinhole,1,1,3).*reshape([1,omega,omega^2],1,1,3);

        %% Capture images
        for img_i = 1:3
            if img_i == 1 && i ~= 1
                continue; % |u| only need to capture once
            end
            img_Ui = abs(OM.crop(iFourier(GT_Fwav.*masks(:,:,img_i)))).^2;
            if OM.contain_noise  
                [img_Ui,SNR] = OM.addnoise(img_Ui);                    
                fprintf('SNR = %.3f\n', SNR);                    
            end
            Mimgs(:,:,img_i) = img_Ui;
        end

        %% Derive r
        ur = mean(Mimgs.*reshape([1,omega,omega^2],1,1,3),3);  
        r_wav = OM.crop(iFourier(pinhole));
       
        %% Derive u    
        u   = ur.* r_wav;           
        u   = sqrt(Mimgs(:,:,1)).*exp(1j*angle(u));    

        %% Merge u from several iterations
        if i == 1
            u0              = u;
            u_sum           = u;
        else
            [~,phase_cor]   = wav_loss(u0, u);
            u_sum           = u_sum + u.*exp(1j* phase_cor);    
        end
                
        est_wav = u_sum/i;      
        est_wav = sqrt(Mimgs(:,:,1)).*exp(1j*angle(est_wav));    

        %% Visualization
        figure(1);
        err = OM.evaluate_result(GT_wav,est_wav);        
        fprintf('iter %d, err = %f, num_of_meas= %d\n' ,i, err, iter_num*2+2);     
        err_all(i) = err;
    end
end

