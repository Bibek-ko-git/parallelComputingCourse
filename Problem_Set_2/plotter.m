% Simple Matlab script to plot the results from heat1d
clear; clf;

% Load the data file generatred by te the heat1d.C

filename = 'heatBcNNNx40.m'; % Change this to your filename
run(filename);

% plots the interior points only 
x = x(2:end-1);
u = u(2:end-1);
err = err(2:end-1);
maxErr = max(err);
fprintf('Max error at t=%f is %e\n', t, maxErr);

% Lets create the plot using the tiled layout 
figure('Position', [100, 100, 1200, 600]);
tiledlayout(1,2);

% Plot the solution
nexttile;
plot(x,u,'-bo','LineWidth',1.5,'MarkerSize',3);
title(['1D Heat Equation: ', solutionName, ' BCs, Nx=', num2str(Nx)]);
xlabel('x'); ylabel('u(x,t)');
grid on;

% Plot the error
nexttile;
plot(x,err,'-ro','LineWidth',1.5,'MarkerSize',3);
title(['Error at t=', num2str(t), ', max error = ', num2str(maxErr)]);
xlabel('x'); ylabel('Error');
grid on;

% overall title for the figure
sgtitle(['1D Heat Equation Solution with ', solutionName, ' Boundary Conditions']);

% Show the plots
% is not working in cgi environment
set(gcf, 'Visible', 'on');
% then save as a png file
saveas(gcf, ['heat1d_', solutionName, 'Nx', num2str(Nx), '.png']);
% End of file
