function stop = stopIfNotImproving(info, patience)
    % Early stopping function based on validation metrics
    stop = false;
    
    % Initialize persistent variables for tracking
    persistent bestMetric;
    persistent numEpochsWithNoImprovement;
    persistent metricName;
    
    if info.State == "start"
        bestMetric = -Inf;
        numEpochsWithNoImprovement = 0;
        
        % Determine which metric to monitor based on the training
        if isfield(info, 'ValidationAccuracy')
            metricName = 'ValidationAccuracy';
        else
            metricName = 'ValidationRMSE';
            bestMetric = Inf; % For RMSE, lower is better
        end
    elseif isfield(info, metricName) && ~isempty(info.(metricName))
        currentMetric = info.(metricName)(end);
        
        if strcmp(metricName, 'ValidationAccuracy')
            % For accuracy, higher is better
            if currentMetric > bestMetric
                bestMetric = currentMetric;
                numEpochsWithNoImprovement = 0;
            else
                numEpochsWithNoImprovement = numEpochsWithNoImprovement + 1;
            end
        else
            % For RMSE, lower is better
            if currentMetric < bestMetric
                bestMetric = currentMetric;
                numEpochsWithNoImprovement = 0;
            else
                numEpochsWithNoImprovement = numEpochsWithNoImprovement + 1;
            end
        end
        
        if numEpochsWithNoImprovement >= patience
            stop = true;
            disp(['Early stopping triggered. No improvement for ' num2str(patience) ' epochs.']);
        end
    end
end

