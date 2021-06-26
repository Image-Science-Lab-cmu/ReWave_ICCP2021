function est_wav = WISH(GT_wav, image_num, OM)
    z               = 25;    
    iter_num        = 100;
    
    mask_num        = image_num;
    masks           = zeros(OM.N,OM.N,mask_num);
    GT_Pimgs        = zeros(OM.PN,OM.PN,mask_num);
    Pimgs           = zeros(OM.PN,OM.PN,mask_num); 
    batch_size      = min(8,mask_num);
    est_wav_batches = cell(1,mask_num/batch_size);
      
    est_wav         = ones(OM.N);
    %% Capture images        
    for i = 1:mask_num
        k = OM.N/OM.uprate;
        masks(:,:,i) = OM.upsample(genRandomPhaseMask([k,k]*0.1,[k,k]));   
        GT_Pimgs(:,:,i) = abs(OM.Prop(OM.pad(GT_wav.*masks(:,:,i)),z)).^2;                    
        if OM.contain_noise        
            [Pimgs(:,:,i),SNR] = OM.addnoise( GT_Pimgs(:,:,i));
            fprintf('SNR = %.3f\n', SNR );                           
        else
            Pimgs(:,:,i) = GT_Pimgs(:,:,i);
        end
    end

    %% Optimization      
    for i = 1:iter_num
        est_wav_batch   = repmat(est_wav,1,1,batch_size);        
        for batch_i = 1:mask_num/batch_size
            mask_batch      = masks(:,:,(1:batch_size)+(batch_i-1)*batch_size);    
            Pimgs_batch     = Pimgs(:,:,(1:batch_size)+(batch_i-1)*batch_size);
            est_Pwav_batch  = OM.Prop(OM.pad(est_wav_batch.*mask_batch),z);
            est_Pwav_batch  = sqrt(Pimgs_batch).*exp(1j*angle(est_Pwav_batch ));
            est_wav_batches{batch_i} = OM.crop(OM.Prop(est_Pwav_batch,-z));    
            est_wav_batches{batch_i} = est_wav_batches{batch_i}.*conj(mask_batch);              
        end
        est_wav = squeeze(mean( cat(3,est_wav_batches{:}),3));
        figure(1);
        err = OM.evaluate_result(GT_wav,est_wav);
        fprintf('iter %d, err = %f\n' ,i, err);
    end
end

function padded_pattern = genRandomPhaseMask(res,siz)
    pattern = rand(res)*2*pi;
    padded_pattern = imresize(pattern,siz,'bicubic');    
    padded_pattern = floor(wrapToPi(padded_pattern)*256)/256;
    padded_pattern = exp(1j*padded_pattern);    
end
