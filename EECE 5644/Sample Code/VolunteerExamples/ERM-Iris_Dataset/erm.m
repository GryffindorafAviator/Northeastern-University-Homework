clc
clear
load('iris.mat')

figure(1);
gscatter(sepalLength, petalLength, class, 'rgb', '^v^', [], 'off');

count_setosa = sum(class == "Iris-setosa");
count_versicolor = sum(class == "Iris-versicolor");
count_virginica = sum(class == "Iris-virginica");
total_data = count_setosa + count_versicolor + count_virginica;

%prior probabilities
prior = [ count_setosa/total_data, count_versicolor/total_data, count_virginica/total_data];

setosa_data = [];
versicolor_data = [];
virginica_data = [];

%segregating data based on classes
for i = 1 : total_data
    if class(i) == "Iris-setosa"
        setosa_data = [setosa_data [sepalLength(i); petalLength(i)]];
    end
    
    if class(i) == "Iris-versicolor"
        versicolor_data = [versicolor_data [sepalLength(i); petalLength(i)]];
    end
    
    if class(i) == "Iris-virginica"
        virginica_data = [virginica_data [sepalLength(i); petalLength(i)]];
    end
        
end

%calculating mean and covariance
setosa_data_mean = [ mean(setosa_data(1, :)); mean(setosa_data(2, :)) ];
versicolor_data_mean = [ mean(versicolor_data(1, :)); mean(versicolor_data(2, :)) ];
virginica_data_mean = [ mean(virginica_data(1, :)); mean(virginica_data(2, :)) ];
setosa_data_cov = cov(setosa_data');
versicolor_data_cov = cov(versicolor_data');
virginica_data_cov = cov(virginica_data');
sigma(:,:,1) = setosa_data_cov;
sigma(:,:,2) = versicolor_data_cov;
sigma(:,:,3) = virginica_data_cov;
mu = [setosa_data_mean'; versicolor_data_mean'; virginica_data_mean'];

%liklihood fn
liklihood = @(x,class) mvnpdf(x, mu(class, :), sigma(:,:,class));

%cost matrix
cost = [ 0, 1, 1; 1, 0 ,1; 1, 1, 0];

%expected  cost calculation
R1 = @(x) (cost(1,1) * liklihood(x, 1)*prior(1)) + (cost(1,2)*liklihood(x,2)*prior(2)) + (cost(1,3)*liklihood(x,3)*prior(3));
R2 = @(x) (cost(2,1)*liklihood(x, 1)*prior(1)) + (cost(2,2)*liklihood(x,2)*prior(2)) + (cost(2,3)*liklihood(x,3)*prior(3));
R3 = @(x) (cost(3,1)*liklihood(x, 1)*prior(1)) + (cost(3,2)*liklihood(x,2)*prior(2)) + (cost(3,3)*liklihood(x,3)*prior(3));

%Decision rule
classfy = @(x) min([R1(x),R2(x) ,R3(x) ]);

%finding the regions by brute force
figure(2);
for h = 4:0.1:9
    for k = 0:0.1:8
        [expected_min_risk, label] = classfy([h, k]);
        hold on;
        if(label == 1)
            scatter(h,k, 50,[0.9,0.9, 0.9],'filled');
        end
        if(label == 2)
            scatter(h,k, 50,[0.4,0.4, 0.4],'filled');
        end
        if(label == 3)
            scatter(h,k, 50,[0.7,0.7, 0.7],'filled');
        end
        hold off;   
    end
end


hold on;
gscatter(sepalLength, petalLength, class, 'rgb', '^v^', [], 'off');
%axis equal;
hold off;


