function bin = code2bin(code,threhold)

[dim nSamples] = size(code);
if  (nargin<2)
        threhold = repmat(median(code,2),1,nSamples);
        
else
    
    threhold = repmat(threhold ,1 ,nSamples);
end

bin = code > threhold;

end
        
    
    