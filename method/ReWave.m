function [est_wav, err_all] = ReWave(GT_wav, image_num, GS_finetune, OM)
    %% Param initialization
    iter_num    = floor( (image_num-2)/2);         
    err_all     = zeros(1,iter_num);
    Mimgs       = zeros(OM.N,OM.N,3);

    %% Capture Fourier image  
    GT_Fwav     = Fourier(OM.pad(GT_wav));    
    if OM.contain_noise     
        [Fimg,SNR] = OM.addnoise(abs(GT_Fwav).^2);                    
        fprintf('F SNR = %.3f\n', SNR);                        
    else
        Fimg        = abs(GT_Fwav).^2;        
    end

    Fimg_ds         = OM.downsample(Fimg);   
    total_energy    = sum(Fimg_ds(:));    
    
    for i = 1:iter_num  
        %% Find collision free mask        
        sparse_num = OM.N;               
        if i == 1
            y = Fimg_ds;
        else
            y(support_set) = 0;                
        end   
        [support_set, support_map, sparse_num] = gen_collision_free_mask(y,sparse_num,total_energy);        
        upsampled_support_map = OM.upsample(support_map);                
        fprintf('sparse num = %d\n',sparse_num);     
        
        %% SLM phase masks        
        omega = (-1+sqrt(3)*1j)/2;
        phase_masks = (1-upsampled_support_map) + ...
                      repmat(upsampled_support_map,1,1,3).*reshape([1,omega,omega^2],1,1,3);

        %% Capture images
        figure(2);
        for img_i = 1:3
            if img_i == 1 && i ~= 1
                continue; % |u| only need to capture once
            end
            img_Ui = abs(OM.crop(iFourier(GT_Fwav.*phase_masks(:,:,img_i)))).^2;
            if OM.contain_noise  
                [img_Ui,SNR] = OM.addnoise(img_Ui);                    
                fprintf('SNR = %.3f\n', SNR);                    
            end
            subplot(1,3,img_i);
            OM.drawint(img_Ui);
            Mimgs(:,:,img_i) = img_Ui;
        end
        drawnow;             

        %% Derive |r|^2
        r_GT  = OM.crop(iFourier(GT_Fwav.*upsampled_support_map));                
        alpha = mean(Mimgs,3);  %eq(5); 
        beta  = mean(Mimgs.*reshape([1,omega,omega^2],1,1,3),3); %eq(6);    
        disc  = max(alpha.^2 - 4*abs(beta).^2,0);
        root1 = (alpha+ sqrt(disc))/2;
        root2 = (alpha- sqrt(disc))/2;      
        r_img = root2; %initialize by smaller one in root

        %% Solve r by SGD
        autcr_support_id    = autocorrelation_support(support_set,size(support_map,1));        
        autcr               = ifft2(r_img)*OM.N^4;     
        corr_est            = reshape(autcr(autcr_support_id),sparse_num,sparse_num);
        diag_id             = (1:sparse_num) + (0:sparse_num-1)*sparse_num;
        corr_est(diag_id)   = Fimg_ds(support_set);
        [U,sig,~]           = svd(corr_est);
        xS_est              = sqrt(sig(1,1))*U(:,1);
        
        Frwav               = zeros(OM.N);
        Frwav(support_set)  = xS_est;
        Frwav               = ifftshift(Frwav);
        Frwav               = OM.upsample(Frwav);
        %Frwav = sqrt(ifftshift(Fimg.*upsampled_support_map)).*exp(1j*angle(Frwav));      
        %Paper results include line above to use Fimg, but remove it is actually better
        pad_rwav = ifft2(Frwav);        
        r_wav   = pad_rwav(1:OM.N,1:OM.N);
        r_abs   = abs(r_wav);

        %% Update r_abs with observation        
        r_abs   = find_closest_x(r_abs,sqrt(root1),sqrt(root2));    
        r_wav   = r_abs.*exp(1j*angle(r_wav));
        
        fprintf('r err = %f\n', wav_loss(r_GT,r_wav));  
        
        %% ReWave GS
        if GS_finetune    
            pad_rwav= OM.pad(r_abs.*exp(1j*angle(r_wav)));
            for computing_loop = 1:10
                for GS_loop = 1:30
                    r_wav       = OM.crop(pad_rwav);
                    pad_rwav(OM.range(),OM.range())= r_abs.*exp(1j*angle(r_wav));                                                                                   
                    Frwav       = Fourier(pad_rwav);
                    Frwav       = sqrt(Fimg).*upsampled_support_map.*exp(1j*angle(Frwav));
                    pad_rwav    = iFourier(Frwav);
                end               
                fprintf('inner iter %d, err = %f\n', computing_loop , sum(sum((abs(r_wav)-r_abs).^2)));
                r_abs = abs(OM.crop(pad_rwav));
                r_abs = find_closest_x(r_abs,sqrt(root1),sqrt(root2));                      
            end         
            fprintf('GS fine_tune r_wav err = %f\n' , wav_loss(r_GT,r_wav));        
        end

        %% Derive u    
        ur  = beta + r_abs.^2;
        u   = ur.*r_wav;           
        u   = sqrt(Mimgs(:,:,1)).*exp(1j*angle(u));    

        %% Merge u from several iterations
        if i == 1
            u0              = u;
            r_int_sum       = r_abs.^2;
            weighted_u_sum  = u.*r_abs.^2;
        else
            [~,phase_cor]   = wav_loss(u0, u);
            u               = u.*exp(1j* phase_cor);        
            r_int_sum       = r_int_sum+r_abs.^2;
            weighted_u_sum  = weighted_u_sum + u.*r_abs.^2;    
        end
                
        est_wav = weighted_u_sum./(r_int_sum+10^-6);      
        est_wav = sqrt(Mimgs(:,:,1)).*exp(1j*angle(est_wav));    

        %% Visualization
        figure(1);
        err = OM.evaluate_result(GT_wav,est_wav);        
        fprintf('iter %d, err = %f, num_of_meas= %d\n' ,i, err, iter_num*2+2);     
        err_all(i) = err;
    end
end

function [support_set, support_map, sparse_num] = gen_collision_free_mask(y,sparse_num,total_energy)                      
    base_rate = 0.05;
    cum_energy = 0; 
    [max_point,sid_n] = max(y(:)); 
    cum_energy =  cum_energy + max_point;                
    y(sid_n) = 0;
    support_set = sid_n; 
    dist_list = [];
    for sn = 1: sparse_num-1                
        [max_point,sid_n] = max(y(:));
        cum_energy =  cum_energy + max_point;            
        if(cum_energy/total_energy >base_rate || sid_n == 1)
            sparse_num = sn;
            break;
        end                
        dist_list = [dist_list,abs(support_set - sid_n)];
        kn = length(support_set);
        l1 = min(repmat(support_set,1,kn) + repelem(abs(support_set - sid_n),1,kn), length(y(:)));
        l2 = max(repmat(support_set,1,kn) - repelem(abs(support_set - sid_n),1,kn), 1);
        l3 = min(sid_n + dist_list,  length(y(:)));
        l4 = max(sid_n - dist_list,  1);                        
        l5 = support_set+sid_n; %middle point    
        l5 = l5(mod(l5,2)== 0);
        l5 = floor(l5/2);
        y([l1,l2,l3,l4,l5]) = 0;
        support_set = [support_set,sid_n];
    end
    support_map  = zeros(size(y));              
    support_map(support_set) = 1;         
end

function autcr_support_id = autocorrelation_support(support_id,N)
    %support_id: 1-dimension id in the support set of a 2-dimension matrix
    %N: width of the matrix
    
    alpha_all           = ceil(support_id/N); 
    beta_all            = mod(support_id,N); 
    beta_all( beta_all== 0) = N;
    alpha_all_pad       = mod(repelem(alpha_all,1,length(alpha_all))-repmat(alpha_all,1,length(alpha_all)),N)+1;
    beta_all_pad        = mod(repelem(beta_all,1,length(beta_all))-repmat(beta_all,1,length(beta_all)),N)+1;               
    autcr_support_id    = beta_all_pad + (alpha_all_pad-1)*N;    
end

function x = find_closest_x(x,x1,x2)
    d1 = abs(x1-x );
    d2 = abs(x2-x );
    x = (d1<d2).*x1 + (d1>=d2).*x2;    
end
