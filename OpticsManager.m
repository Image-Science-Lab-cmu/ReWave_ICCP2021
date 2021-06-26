classdef OpticsManager < handle
    properties
        %assume the camera has the same pitch as the wave
        %assume the SLM has coarser pitch than the wave        
        %wave would be padded accrodingly
        
        dx;     %pixel width
        uprate; %SLM pixel is simulated at dx*uprate  
        N;      %width of image        
        PN;     %width of padded image            
        lambda; %wave length
        contain_noise;
        target_SNR;       

        range;  %range of valid part in padded image 
    end
    methods 
        function om = OpticsManager(params)
            om.dx       = params.dx;                
            om.uprate   = params.uprate;            
            om.N        = params.N ;
            om.PN       = params.PN; 
            om.lambda   = params.lambda;
            om.contain_noise = params.contain_noise;
            om.target_SNR = params.target_SNR;    
            
            om.range    = @() om.PN/2+(-om.N/2+1:(om.N/2));
        end
        
%% For propagation  
        function Fwav = Prop(om,wav,z) %z_rate: z/No 
            az = abs(z);                      
            %critical sampling: N*dx = lambda*z/dx            
            impulse_response = (az >= om.dx^2*om.PN/om.lambda);
            
            if  impulse_response 
                x = -om.PN/2:(om.PN/2-1);
                [X,Y] = meshgrid(x,x);
                h = 1/(1j*om.lambda*az)*exp(1j*pi/om.lambda/az*om.dx^2*(X.^2+Y.^2)); %ignore lambda*z                
                H = fft2(fftshift(h))/(om.PN)^2; %create trans func    
                if(z<0)
                    H = conj(H);
                end                
            else
                fx = 1/(om.dx)*( -1/2:1/om.PN:(1/2-1/om.PN));
                [FX,FY] = meshgrid(fx,fx);
                H = exp(1j*2*pi*z*sqrt(1/om.lambda^2-FX.^2-FY.^2)); 
                H = fftshift(H);
                H= H*conj(H(1,1));                                     
            end
                
            Fwav = ifftshift(ifft2(fft2(fftshift(wav)).*H));
        end

%% For coefficient conversion
        function phase = coef2phase_fourier(om,coefs)
            pad_coefs = zeros(om.N);
            if(mod(size(coefs,1),2) == 1)
                coef_range = om.N/2+1+( -(size(coefs,1)-1)/2:(size(coefs,1)-1)/2);
            else
                coef_range = om.N/2+1+( -size(coefs,1)/2:(size(coefs,1)/2-1));                   
            end
            pad_coefs(coef_range,coef_range) = coefs;
            phase = fftshift(ifft2(ifftshift(pad_coefs)))*om.N^2;
            phase = wrapToPi(real(phase));
        end
        
        function [GT_phase,GT_wav] = gen_simwave(om,test_pattern)
            source_list = {'fourier_coef','zernike'};            
            if (test_pattern <= 8)
                wave_source = source_list{2};
            else
                wave_source = source_list{1};
            end

            switch test_pattern
                case 1
                    zp = struct('n',2,'m',0,'k',9); zp_all = {zp};
                case 2
                    zp = struct('n',2,'m',0,'k',-9); zp_all = {zp};
                case 3
                    zp = struct('n',2,'m',2,'k',27); zp_all = {zp};
                case 4    
                    zp = struct('n',3,'m',1,'k',16); zp_all = {zp};
                case 5
                    zp = struct('n',3,'m',3,'k',16); zp_all = {zp};
                case 6
                    zp_all = {struct('n',3,'m',3,'k',16),struct('n',2,'m',0,'k',-9)};  
                case 7
                    zp_all = {struct('n',2,'m',2,'k',27),struct('n',2,'m',0,'k',-9)}; 
                case 8
                    zp_all = {struct('n',1,'m',-1,'k',9),struct('n',2,'m',0,'k',9),struct('n',3,'m',1,'k',-16)};    
                case 9
                    wav_N = 5; K = 3; load('pattern_coef/fourier_N_5_K_3');
                case 10
                    wav_N = 5; K = 6; load('pattern_coef/fourier_N_5_K_6');
                case 11
                    wav_N = 9; K = 3; load('pattern_coef/fourier_N_9_K_3');
                case 12
                    wav_N = 9; K = 6; load('pattern_coef/fourier_N_9_K_6');                        
                case 13
                    wav_N = 17; K = 3; load('pattern_coef/fourier_N_17_K_3');
                case 14
                    wav_N = 17; K = 6; load('pattern_coef/fourier_N_17_K_6');    
            end

            switch wave_source
                case 'fourier_coef'
                    GT_phase = om.coef2phase_fourier(GT_coefs);                
                case 'zernike'
                    GT_phase = zeros(om.N);
                    for zi = 1:length(zp_all)
                        zp = zp_all{zi};
                        x = -0.5:1/om.N:0.5-(1/om.N);
                        [X,Y] = meshgrid(x,x); 
                        [theta,r] = cart2pol(X,Y);
                        z_hres = reshape(zernfun(zp.n,zp.m,r(:),theta(:)),[om.N,om.N]);
                        GT_phase =  GT_phase + pi*z_hres*zp.k;
                    end
            end
            GT_phase = GT_phase-mean(GT_phase(:));
            GT_phase = wrapToPi(GT_phase);
            GT_wav   = ones(om.N).*exp(1j*GT_phase);
        end
                     
%% For visualization     
        function drawphase(varargin)
            om  = varargin{1};
            phs = varargin{2};            
            phs = om.normalize_phase(real(phs));     
            x_tick = (-om.N/2:(om.N/2-1));           
            imagesc(x_tick,x_tick,phs );    
            xlabel('x'); ylabel('y'); 
            axis xy
            colorbar;
            colormap(gca,'hsv');
            if(nargin >2)
                caxis(varargin{3});                                        
            end            
        end
        function drawint(om,intensity)
            imagesc(intensity);
            xlabel('x'); ylabel('y');
            %title(tt);
            axis xy
            colorbar;
            colormap(gca,'gray');
            caxis([0 2]);
        end
        
        function err = evaluate_result(om,GT_wav,est_wav)
            GT_phase = angle(GT_wav);
            est_phase = angle(est_wav);
            
            subplot(1,3,1);
            om.drawphase(GT_phase,[-pi,pi]);
            title('GT phase');            
                        
            est_wav = sqrt(abs(GT_wav)).*exp(1j*est_phase); %only compare error of phase
            [err,phase_cor] = wav_loss(GT_wav,est_wav);

            est_phase = est_phase + phase_cor;

            subplot(1,3,2);   
            om.drawphase(est_phase,[-pi,pi]);        
            title('est phase');
            subplot(1,3,3);
            phase_diff = GT_phase-est_phase;
            om.drawphase(phase_diff);            
            title(sprintf('Rel err = %f', err));    
            drawnow;            
        end
        
%% For noise            
        function [noiseimg,SNR] = addnoise(om,img)
            %assume exposure adjust the max image value = 0.9
            normalization_rate = 0.9/max(img(:));
            normalized_img = img*normalization_rate;
                        
            F = 30500;
            x = 10^(om.target_SNR/10 );   
            R = 71.7;
            sig = F*10^-(R/20);                      
            G = F/(x/2+sqrt(x^2+x*sig^2)/2)/3;
            %SNR is controled by giving different gain G to the image
            %G is estimated by the target SNR 
            %assume image is roughly a uniform image with value ~ 1/3
            
            noiseimg = (poissrnd(normalized_img/G*F) + sig*randn(size(normalized_img)));                        
            noiseimg = noiseimg/F*G;
            noiseimg = max(min(noiseimg,1),0);
            noiseimg = noiseimg/normalization_rate;
            noise = noiseimg-img;
            SNR = sum(sum(img.^2))/sum(sum(noise.^2));
            SNR = 10*log10(SNR);
        end        

%% For utility
        function wav_us = upsample(om,wav)
             wav_us = imresize(wav,size(wav)*om.uprate,'nearest');
        end
        function wav_ds = downsample(om,wav)
            rate = om.uprate;             
            did = (rate+1)/2:rate:size(wav,1);
            
            if(length(size(wav)) == 2)
                wav_ds  = conv2(wav ,ones(rate)/rate^2,'same');  
                wav_ds  = wav_ds( did,did); 
            elseif(length(size(wav)) == 3)
                wav_ds  = convn(wav ,ones(rate,rate,1)/rate^2,'same');  
                wav_ds  = wav_ds( did,did,:); 
            end
        end
        function cropped_wav = crop(om,wav)
            if length(size(wav)) == 3
                cropped_wav = wav(om.range(),om.range(),:);                
            else
                cropped_wav = wav(om.range(),om.range());
            end
        end
        function pad_wav = pad(om,wav)
            if length(size(wav)) == 3
                pad_wav = zeros(om.PN,om.PN,size(wav,3));                
                pad_wav(om.range(),om.range(),:) = wav;                
            else
                pad_wav = zeros(om.PN,om.PN);            
                pad_wav(om.range(),om.range()) = wav;
            end
        end
     
        function phase = normalize_phase(om,phase)
            phase = wrapToPi(phase);
            phase = phase - mean(phase(:));
            phase = wrapToPi(phase);
        end  
    end
end

