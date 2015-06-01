function [EuclideanDistance,sortIndex] = EuclidDist(onecode,code,hsortIndex);


if nargin ==3 
    wishcode = code(:,hsortIndex);

    [dim nSamples] = size(wishcode);

    EuclideanDistance = zeros(1,nSamples);

    for ii=1:nSamples
        EuclideanDistance(ii) = norm(onecode-wishcode(:,ii));
    end

    [EuclideanDistance,sortIndex] = sort(EuclideanDistance);

    sortIndex = hsortIndex(sortIndex);

else
    
    for ii=1:size(code,2)
    EuclideanDistance(ii) = norm(onecode-code(:,ii));
    end

    [EuclideanDistance,sortIndex] = sort(EuclideanDistance);
    
end
