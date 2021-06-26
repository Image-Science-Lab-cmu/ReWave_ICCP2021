addpath('./method');
addpath('./util');

close all;
clear;

%method_list = {'GS','MP','WISH','PDI','ReWave','ReWave_GS'};
method              = 'ReWave_GS';
test_pattern_id     = 14;%1~14
image_num           = 32;

params = struct('uprate',3,'N',150,'dx',2.13e-3,'lambda',560e-6, ...
    'contain_noise',true,'target_SNR',60);

%wave/camera pitch: dx
%SLM pitch: dx*uprate

if find(strcmp( {'MP','WISH'},method))
    params.PN = 1032; %critical sampling rate at SLM pitch: L = PN*dx = lambda*z/(dx*uprate)
else
    params.PN = params.N*params.uprate; 
end

OM = OpticsManager(params);           

[GT_phase, GT_wav] = OM.gen_simwave(test_pattern_id);
est_wav = ones(OM.N);           
switch method
    case 'GS'
        est_wav = Gerchberg_Saxton(GT_wav,image_num,OM);  
    case 'MP'
        est_wav = Multiplane_Propagation(GT_wav,image_num,OM);                  
    case 'WISH'
        est_wav = WISH(GT_wav,image_num,OM);                                  
    case 'PDI'
        [est_wav,err_all_iter] = PDI(GT_wav, image_num, OM);                                                             
    case 'ReWave'
        [est_wav,err_all_iter] = ReWave(GT_wav, image_num, false, OM);                                         
    case 'ReWave_GS'
        [est_wav,err_all_iter] = ReWave(GT_wav, image_num, true, OM);                                         
end  

err = OM.evaluate_result(GT_wav,est_wav);            
