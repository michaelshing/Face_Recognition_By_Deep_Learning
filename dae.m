% dae - training a single-layer DAE
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
function [D] = dae(D, patches,vsigmoid,hsigmoid)


n_samples = size(patches, 1);

if D.structure.n_visible ~= size(patches, 2)
    error('Data is not properly aligned');
end


minibatch_sz = D.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);
batchloop_times = D.batchloop_times;


weight_decay = D.learning.weight_decay;

n_visible = D.structure.n_visible;
n_hidden = D.structure.n_hidden;

do_normalize = D.do_normalize;
do_normalize_std = D.do_normalize_std;


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


theta = [D.W1(:); D.W2(:); D.hbias(:); D.vbias(:)];  
                        

options.Method = D.optimize.Method;
options.maxIter = D.optimize.maxIter;
options.corr = D.optimize.corr;

beta =  D.sparsity.cost;  
sparsityParam = D.sparsity.target;
finalObjective = D.finalObjective;
early_stop  = D.early_stop;

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

        if  D.noise.level > 0
            v0 = v0 + D.noise.level * randn(size(v0));
        end

        if D.noise.drop > 0
            mask = binornd(1, 1 - D.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end
                        
        v0 = v0';
        v0_clean = v0_clean';

        [theta, loss] = minFunc( @(p) AutoencoderLoss(p, ...
                        n_hidden, n_visible, weight_decay, sparsityParam, beta, ... 
                        v0, v0_clean, vsigmoid, hsigmoid), theta, options);
               
        if early_stop == 1
          
          if loss <= finalObjective % use the minibatch obj as a heuristic for stopping
                                 % because checking the entire dataset is very
                                 % expensive
            % yes, we should check the objective for the entire training set        
            trainError = AutoencoderLoss(theta, n_hidden, n_visible, weight_decay,...
                         sparsityParam, beta, v0, v0_clean,vsigmoid,hsigmoid);
            if trainError <= finalObjective
                % now your submission is qualifieds
                break
            end
          end  
        end
end

D.W1 = reshape(theta(1:n_hidden*n_visible), n_hidden, n_visible);
D.W2 = reshape(theta(n_hidden*n_visible+1:2*n_hidden*n_visible), n_visible, n_hidden);
D.hbias = theta(2*n_hidden*n_visible+1:2*n_hidden*n_visible+n_hidden);
D.vbias = theta(2*n_hidden*n_visible+n_hidden+1:end);

end




        
