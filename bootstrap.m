
function [J,K] = bootstrap(matrix, n, m)
    %myFun - Description
    %
    % Syntax: output = myFun(input)
    %
    % Long description
    
    J = zeros(m, length(matrix(1, :)));
    K = zeros(m, length(matrix(1, :)));
    
    for i = 1:m
        random = randi(n, 1, n);
        A = matrix(random, :);
        J(i, :) = sum(A);
        K(i, :) = sum(A.^2);
    end
end
    