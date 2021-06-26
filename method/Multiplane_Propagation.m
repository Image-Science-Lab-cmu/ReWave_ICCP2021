function est_wav = Multiplane_Propagation(GT_wav, image_num, OM)
    iter_num = 50; %run forward and backward in one iter   
    plane_num = image_num;
    
    est_wav = ones(OM.N);
    d1 = 10;
    d2 = 25;
    dz = (d2-d1)/(plane_num-1);
    plane_dist = d1:dz:d2; 
    Pimgs = cell(1,plane_num);
    
    %% Capture images    
    for i = 1:plane_num
        if i == 1
            GT_Pwav = OM.Prop(OM.pad(GT_wav),plane_dist(i));
        else
            GT_Pwav = OM.Prop(GT_Pwav,plane_dist(i)-plane_dist(i-1));            
        end
        
        if OM.contain_noise
            [Pimgs{i},SNR] = OM.addnoise(abs(GT_Pwav).^2);
            fprintf('SNR = %.3f\n', SNR );                           
        else
            Pimgs{i} = abs(GT_Pwav).^2;                       
        end

    end
    
    est_Pwav = OM.Prop(OM.pad(est_wav),plane_dist(1));
    est_Pwav = Pimgs{1}.*exp(1j*angle(est_Pwav));    

    %% Optimization      
    for i = 1:iter_num 
        for plane_i = 2:plane_num
            est_Pwav = OM.Prop(est_Pwav,plane_dist(plane_i)-plane_dist(plane_i-1));
            est_Pwav = sqrt(Pimgs{plane_i}).*exp(1j*angle(est_Pwav));
        end

        for plane_i = plane_num:-1:2
            est_Pwav = OM.Prop(est_Pwav,plane_dist(plane_i-1)-plane_dist(plane_i));                 
            est_Pwav = sqrt(Pimgs{plane_i-1}).*exp(1j*angle(est_Pwav));                
        end
        est_Pwav = sqrt(Pimgs{1}).*exp(1j*angle(est_Pwav));            
        est_wav = OM.crop(OM.Prop(est_Pwav,-plane_dist(1)));
        figure(1);
        err = OM.evaluate_result(GT_wav,est_wav);
        fprintf('iter %d, err = %f\n' ,i, err);
    end    
end