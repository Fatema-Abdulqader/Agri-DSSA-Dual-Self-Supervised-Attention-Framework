%% Helper Functions
%
function [trainIdx, valIdx, testIdx] = stratifiedSplit(labels, ratios)
    % Perform stratified sampling based on class labels
    classes = unique(labels);
    trainIdx = [];
    valIdx = [];
    testIdx = [];
    
    for c = 1:length(classes)
        % Find indices of current class
        idx = find(labels == classes(c));
        
        % Shuffle indices
        idx = idx(randperm(length(idx)));
        
        % Calculate split sizes
        nTrain = round(length(idx) * ratios(1));
        nVal = round(length(idx) * ratios(2));
        
        % Split indices
        trainIdx = [trainIdx; idx(1:nTrain)];
        valIdx = [valIdx; idx(nTrain+1:nTrain+nVal)];
        testIdx = [testIdx; idx(nTrain+nVal+1:end)];
    end
    
    % Shuffle again
    trainIdx = trainIdx(randperm(length(trainIdx)));
    valIdx = valIdx(randperm(length(valIdx)));
    testIdx = testIdx(randperm(length(testIdx)));
end

