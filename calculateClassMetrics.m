function [precision, recall, f1score] = calculateClassMetrics(trueLabels, predLabels)
    % Calculate precision, recall, and F1-score for each class
    classes = categories(trueLabels);
    numClasses = length(classes);
    
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1score = zeros(numClasses, 1);
    
    for c = 1:numClasses
        % True positives, false positives, false negatives
        tp = sum(predLabels == classes(c) & trueLabels == classes(c));
        fp = sum(predLabels == classes(c) & trueLabels ~= classes(c));
        fn = sum(predLabels ~= classes(c) & trueLabels == classes(c));
        
        % Precision, recall, F1 score
        if tp + fp > 0
            precision(c) = tp / (tp + fp);
        else
            precision(c) = 0;
        end
        
        if tp + fn > 0
            recall(c) = tp / (tp + fn);
        else
            recall(c) = 0;
        end
        
        if precision(c) + recall(c) > 0
            f1score(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c));
        else
            f1score(c) = 0;
        end
    end
end

