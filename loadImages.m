function images = loadImages(dirname,imgtype)

subdir=dir(dirname);
images=[];
x = 64;
y = 64;
counts=0;
for ii=1:length(subdir)     
    
    subname=subdir(ii).name;
	if ~strcmp(subname, '.') && ~strcmp(subname, '..')
        frames = dir(fullfile(dirname, subname, imgtype));
        for jj=1:length(frames); 
            counts = counts+1;
            if mod(counts,1000)==0
                fprintf('img NO: %d\n',counts);
            end
            imgpath = fullfile(dirname, subname, frames(jj).name);
            I=imread(imgpath);
            if ndims(I)==3
                I=rgb2gray(I);
            end
            I = imresize(I, [x y]);
            I=double(I);
            I=I(:);
            images=[images,I];
        end
    end
end
