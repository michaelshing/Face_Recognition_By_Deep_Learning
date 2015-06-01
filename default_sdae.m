% default_dbm - 
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
function [S] = default_sdae (layers)

    % learning parameters
    S.learning.weight_decay = 3e-3;
    S.learning.minibatch_sz = 30;

    S.do_normalize = 0;
    S.do_normalize_std = 0;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.

    % denoising
    S.noise.drop = 0.1;
    S.noise.level = 0.1;

    % structure
    n_layers = length(layers);
    S.structure.layers = layers;

    % initializations
    S.W1 = cell(n_layers-1, 1);
    S.W2 = cell(n_layers-1, 1);
    S.hbiases = cell(n_layers-1, 1);
    S.vbiases = cell(n_layers-1, 1);
    for l = 1:n_layers-1
        S.hbiases{l} = zeros(layers(l+1), 1);
        S.vbiases{1} = zeros(layers(l), 1);             %add by michael for two way finetune
        
        %S.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
            
        S.W1{l} = 2 * sqrt(6)/sqrt(layers(l)+layers(l+1)+1) * (rand(layers(l+1), layers(l)) - 0.5);
        S.W2{l} = 2 * sqrt(6)/sqrt(layers(l+1)+layers(l)+1) * (rand(layers(l), layers(l+1)) - 0.5);
    end
    
    % optimize options
    S.optimize.Method = 'lbfgs';
    S.optimize.maxIter = 50;
    S.optimize.corr = 10 ;
    
    
    % batch loop times for minibatch
    S.batchloop_times = 2; %should be large than or eauql to 1    % batch loop times for minibatch
    
    % final objective for early-stop , you know this precedure is good for
    % reducing the optimize time and over-fitting
    S.early_stop=1;
    S.finalObjective = 1e+0 ; % ONLY takes effect when early_stop is allowed by me
    
    %display
    S.display = 1;
    
    %layerstype ,sigmoid or linear
    
    S.layerstype = ones(1,2*n_layers-1);
    
    
    %NCA
    
    S.ncaOBJ = 0 ;
    
    
end
