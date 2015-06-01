% NCA
% Copyright (C) 2013 Michael Shing 
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


function S = nca(data, label, S)

layers = S.structure.layers;
n_layers = length(layers);
layerstype = S.layerstype(1:n_layers);
theta = [];
for tt=1:n_layers-1
    theta = [theta; S.W1{tt}(:); S.hbiases{tt}(:)];
end

n_samples = size(data, 2);
minibatch_sz = S.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);
batchloop_times = S.batchloop_times;
weight_decay = S.learning.weight_decay;
ncaOBJ= S.ncaOBJ;
early_stop= S.early_stop;  % may be should be 0


options.Method = S.optimize.Method;
options.maxIter = S.optimize.maxIter;
options.corr = S.optimize.corr;

for ii=1:batchloop_times*n_minibatches
    
        if ii>n_minibatches;

            mb=mod(ii,n_minibatches);

            if mb==0 
                fprintf('another LOOP...\n\n');
                newIndex = randperm(n_samples);
                data=data(:,newIndex);
                label=label(newIndex);
                mb=1;
            end
        else
            mb=ii;
        end
        
        subData = data(:,(mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples));
        subLabel = label((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples));

        [theta, loss] = minFunc( @(p) ncaLoss (p, layers, layerstype ,weight_decay, subData , subLabel), theta, options);
         
        if early_stop == 1
          
          if loss <= ncaOBJ % use the minibatch obj as a heuristic for stopping
                                 % because checking the entire dataset is very
                                 % expensive
            % yes, we should check the objective for the entire training set        
            COST = ncaLoss (theta, layers, layerstype, weight_decay,  subData, subLabel);
            if COST <= ncaOBJ
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
 
 
 
 
 
 
 






end