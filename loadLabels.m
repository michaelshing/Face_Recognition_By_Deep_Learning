function labels = loadLabels(dirname,imgtype)

subdir=dir(dirname);
labels=[];
class=0;
for ii=1:length(subdir)     
    
    subname=subdir(ii).name;
	if ~strcmp(subname, '.') && ~strcmp(subname, '..')
        class=class+1;
        frames = dir(fullfile(dirname, subname, imgtype));
        tmp=zeros(length(frames),1); 
        tmp(:)=class;
        labels=[labels;tmp(:)];
        end
    end
end
