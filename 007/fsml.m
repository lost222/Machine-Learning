function [ W, bt, Y_train ] = fsml( X_train, Y_train, param )
%%A Convex Formulation for Semi-Supervised Multi-Label Feature Selection
%%X_train: Training data, including Labeled training data and Unlabeled training data
%%Y_train: Training data label
%%param: parameters

%% Reference:
%% Xiaojun Chang, Feiping Nie, Yi Yang and Heng Huang. A Convex Formulation for Semi-Supervised Multi-Label Feature Selection. AAAI 2014.

alpha = param.alpha;
% dim row n col
[dim, n] = size(X_train);

% lable_num == Y_train.col
label_num = size(Y_train, 2);

% sum Y 按照列加起来
% find 找到其中非零元素 ---- 至少有一个lable的实例
% 给出这些标签是不是0 ？ 这有什么意义？
% labled_num 是 Y 中有多少实例是至少有一个label的

labeled_id = (find(sum(Y_train, 2) ~= 0));
labeled_num = length(labeled_id);
%Initialize W
W = rand(dim, label_num);

iter = 1;
obji = 1;

% eye(n) 返回一个 Rank(n)的单位矩阵
H = eye(n) - 1 / n * ones(n, 1) * ones(n, 1)';

while 1
    
    d = 0.5./sqrt(sum(W.*W, 2) + eps);
    D = diag(d);
    W = (X_train * H * H * X_train' + 2 * alpha * D + eps) \ X_train * H * Y_train;
    bt = 1/n * ( ones(n, 1)' * Y_train - ones(n, 1)' * X_train' * W);
    Ypred = X_train' * W + ones(n, 1) * bt;
    for i = labeled_num+1:n
        for j = 1:label_num
           if Ypred(i, j) <= 0
               Y_train(i, j) = 0;
           else
               if Ypred(i, j) >= 1
                   Y_train(i, j) = 1;
               else
                   Y_train(i, j) = Ypred(i ,j);
               end
           end
        end
    end
    
    objective(iter) = (norm((X_train'*W + ones(n,1)*bt - Y_train), 'fro'))^2 + alpha * sum(sqrt(sum(W.*W,2)+eps));
    cver = abs((objective(iter) - obji)/obji);
    obji = objective(iter);
    iter = iter + 1;
    if (cver < 10^-3 && iter > 2) , break, end
        
end




end

