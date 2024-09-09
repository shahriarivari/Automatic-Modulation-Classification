function plot_example_signal(examples_dict, modulationType, SNR_str, exampleIdx)
    % Check if the specified key exists in the dictionary
    key = sprintf('%s__%s', modulationType, SNR_str);
    if ~isfield(examples_dict, key)
        error('The specified modulation type and SNR combination does not exist.');
    end

    % Extract the signal
    signal = examples_dict.(key)(exampleIdx, :, :, :);
    signal = squeeze(signal(1,1,:,:));
    

    % Plot the constellation
    figure;
    scatter(signal(1,:), signal(2,:), 'filled');
    title(sprintf('Constellation Diagram for %s at SNR = %s (Example %d)', modulationType, SNR_str, exampleIdx));
    xlabel('In-Phase');
    ylabel('Quadrature');
    grid on;
end
