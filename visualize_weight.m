% visualize_rbm - Visualize 
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
function [f] = visualize_dae(R, l, opt)

    n_visible = floor(sqrt(size(R,1))).^2;
    titlename=fullfile(opt,['layer_', num2str(l)], [num2str(size(R,1)),'x', num2str(size(R,2))]);
    figure;
    visualize(R(1:n_visible,:), 1);
    title (titlename);
    axis image off
    drawnow;

end

 