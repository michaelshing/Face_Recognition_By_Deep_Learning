% sdae_get_hidden
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [h_mf] = sdae_get_hidden(x0, S)

layers = S.structure.layers;
n_layers = length(layers);
layerstype= S.layerstype(1:n_layers);
h_mf = x0;

for l = 2:n_layers
    
    h_mf = bsxfun(@plus, S.W1{l-1}* h_mf, S.hbiases{l-1});
    
    if layerstype(l)
        h_mf = sigmoid(h_mf);
    end
end


