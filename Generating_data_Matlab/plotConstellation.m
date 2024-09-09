% Function to plot constellation
function plotConstellation(signal, titleStr)
    scatterplot(signal);
    title(titleStr);
    xlabel('In-Phase');
    ylabel('Quadrature');
    grid on;

end