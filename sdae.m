% sdae - training a stacked DAE (finetuning)
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [S] = sdae(S, patches,blayers)


if nargin < 3
    early_stop = 0;
else
    early_stop = 1;
end

n_samples = size(patches, 1);

layers = S.structure.layers;
n_layers = length(layers);

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = S.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

weight_decay = S.learning.weight_decay;

do_normalize = S.do_normalize;
do_normalize_std = S.do_normalize_std;


if do_normalize == 1
    % make it zero-mean
    patches_mean = mean(patches, 1);
    patches = bsxfun(@minus, patches, patches_mean);
end

if do_normalize_std ==1
    % make it unit-variance
    patches_std = std(patches, [], 1);
    patches = bsxfun(@rdivide, patches, patches_std);
end


theta=[];

for tt=1:n_layers-1
    theta = [theta; S.W1{tt}(:); S.hbiases{tt}(:)];
end

for tt=n_layers-1:-1:1
    theta = [theta; S.W2{tt}(:); S.vbiases{tt}(:)];
end


flip_layers = fliplr(layers);
netconfig = [ layers, flip_layers(2:end)]; % 384-200-200-100-200-200-384 added by michael

flip_blayers =  fliplr(blayers);
layerstype = [blayers,flip_blayers(2:end)];
beta = 0;

options.Method = S.optimize.Method;
options.maxIter = S.optimize.maxIter;
options.corr = S.optimize.corr;

early_stop = S.early_stop;
finalObjective =  S.finalObjective;
S.layerstype = layerstype;
batchloop_times=S.batchloop_times;

for ii=1:batchloop_times*n_minibatches
    
        if ii>n_minibatches;
            mb=mod(ii,n_minibatches);
            if mb==0
                mb=1;
            end
        else
            mb=ii;
        end
        
        % p_0
        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);

        % add error
        v0_clean = v0;

        if  S.noise.level > 0
            v0 = v0 + S.noise.level * randn(size(v0));
        end

        if S.noise.drop > 0
            mask = binornd(1, 1 - S.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end
        
        v0 = v0';
        v0_clean = v0_clean';
        [theta, loss] = minFunc( @(p) stackedAutoencoderLoss(p, ...
                      netconfig, weight_decay, beta, ... 
                      v0, v0_clean, S.layerstype), theta, options);
        
        if early_stop == 1
          
          if loss <= finalObjective % use the minibatch obj as a heuristic for stopping
                                 % because checking the entire dataset is very
                                 % expensive
            % yes, we should check the objective for the entire training set        
            trainError = stackedAutoencoderLoss(theta, netconfig, weight_decay, beta, ... 
                         v0, v0_clean,S.layerstype);
            if trainError <= finalObjectives
                % now your submission is qualifieds
                break
            end
          end  
       end
                      
 end

 pos=0;
 
 for tt=1:n_layers-1
    wlen = layers(tt) * layers(tt+1);
    S.W1{tt} = reshape(theta(pos+1:pos+wlen), layers(tt+1), layers(tt));
    pos=pos + wlen;
    blen = layers(tt+1);
    S.hbiases{tt}= theta(pos+1:pos+layers(tt+1));
    pos=pos + blen;
 end
 
 for tt=1:n_layers-1
    wlen = flip_layers(tt) * flip_layers(tt+1);
    S.W2{tt} = reshape(theta(pos+1:pos+wlen), flip_layers(tt+1), flip_layers(tt));
    pos=pos + wlen;
    blen = flip_layers(tt+1);
    S.vbiases{tt} = theta(pos+1:pos+flip_layers(tt+1));
    pos=pos + blen;
 end
 
end
 

 
 
 
 


