function modifiedLgraph = removeSpectralAttention(lgraph)
    % Create a version of the model without spectral attention
    modifiedLgraph = lgraph.copy();
    
    % Bypass spectral attention
    try
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'relu_3', 'gap');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'sigmoid', 'spectral_attention/in1');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'relu_3', 'spectral_attention/in2');
        
        % Connect relu_3 directly to spatial_conv
        modifiedLgraph = connectLayers(modifiedLgraph, 'relu_3', 'spatial_conv');
        modifiedLgraph = connectLayers(modifiedLgraph, 'relu_3', 'spatial_attention/in2');
    catch
        warning('Could not modify spectral attention layers. Using original graph.');
    end
    
    return;
end

function modifiedLgraph = removeSpatialAttention(lgraph)
    % Create a version of the model without spatial attention
    modifiedLgraph = lgraph.copy();
    
    % Bypass spatial attention
    try
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spectral_attention', 'spatial_conv');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spatial_sigmoid', 'spatial_attention/in1');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spectral_attention', 'spatial_attention/in2');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spatial_attention', 'final_gap');
        
        % Connect spectral_attention directly to final_gap
        modifiedLgraph = connectLayers(modifiedLgraph, 'spectral_attention', 'final_gap');
    catch
        warning('Could not modify spatial attention layers. Using original graph.');
    end
    
    return;
end

