clear all, close all,

filenames{1,1} = '3096_gray.jpg';
filenames{1,2} = '42049_gray.jpg';
filenames{1,3} = '3096_color.jpg';
filenames{1,4} = '42049_color.jpg';

Kvalues = [2,3,4]; % desired numbers of clusters

for imageCounter = 1:4 %size(filenames,2)
    imdata = imread(filenames{1,imageCounter}); 
    figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1), imshow(imdata);
    if length(size(imdata))==2 % grayscale image
        [R,C] = size(imdata); N = R*C; imdata = double(imdata); % overwriting, since I don't need the uint8 format anymore
        rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
        features = [rowIndices(:)';colIndices(:)';imdata(:)']; 
        minf = min(features,[],2); maxf = max(features,[],2); ranges = maxf-minf;
        x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    elseif length(size(imdata))==3 % color image with RGB color values
        [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
        rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
        features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
        for d = 1:D
            imdatad = imdata(:,:,d); % pick one color at a time
            features = [features;imdatad(:)'];
        end
        minf = min(features,[],2); maxf = max(features,[],2);
        ranges = maxf-minf;
        x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    end
    d = size(x,1); % feature dimensionality
    % K-means clustering
    for k = 1:length(Kvalues)
        K = Kvalues(k); % number of clusters
        centroids = rand(d,K); converged = 0;
        SampleDistancesToCentroids = pdist2(centroids',x','euclidean');
        [~,labels] = min(SampleDistancesToCentroids,[],1);
        while ~converged
            for l = 1:K
                centroids(:,l) = mean(x(:,labels==l),2);
            end
            SampleDistancesToCentroids = pdist2(centroids',x','euclidean');
            [~,newlabels] = min(SampleDistancesToCentroids,[],1);
            converged = isempty(find(newlabels~=labels));
            labels = newlabels; % get ready for next iteration, in case not converged
        end
        labelImage = reshape(labels,R,C);
        figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1+k), imshow(uint8(labelImage*255/Kvalues(k)));
        title(strcat({'Clustering with K = '},num2str(K)));
    end
end
