function modifiedLgraph = removeAllAttention(lgraph)
    % Create a version of the model without any attention
    modifiedLgraph = lgraph.copy();
    
    % Bypass both attention mechanisms
    try
        % Remove spectral attention connections
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'relu_3', 'gap');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'sigmoid', 'spectral_attention/in1');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'relu_3', 'spectral_attention/in2');
        
        % Remove spatial attention connections
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spectral_attention', 'spatial_conv');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spatial_sigmoid', 'spatial_attention/in1');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spectral_attention', 'spatial_attention/in2');
        modifiedLgraph = disconnectLayers(modifiedLgraph, 'spatial_attention', 'final_gap');
        
        % Connect relu_3 directly to final_gap
        modifiedLgraph = connectLayers(modifiedLgraph, 'relu_3', 'final_gap');
    catch
        warning('Could not modify attention layers. Using original graph.');
    end
    
    return;
end