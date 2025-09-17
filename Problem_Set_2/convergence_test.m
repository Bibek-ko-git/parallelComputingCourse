% Simple script to test convergence of the heat1d solver
clear; clc;

% grid resolution setup
Nx_values = [10, 20, 40, 80, 160]; 
numTests = length(Nx_values);
errors_DD = zeros(numTests,1);
errors_NN = zeros(numTests,1);
dx_values = zeros(numTests,1);

fprintf('Running the test for Dirichlet-Dirichlet BC''s\n');
for i = 1:numTests
    Nx = Nx_values(i);
    filename = sprintf('heat1dBcDDNx%d.m',Nx);
    if exist(filename, 'file') ~= 2
        error('File %s does not exist. Please execute heat1d to generate it.', filename);
    else
        run(filename);
        err = err(2:end-1); % interior points only
        maxErr = max(err);
        errors_DD(i) = maxErr;
        dx_values(i) = dx;
        fprintf('Nx=%d, dx=%e, max error=%e\n', Nx, dx, maxErr);
    end
end

fprintf('Running the test for Neumann-Neumann BC''s\n');
for i = 1:numTests
    Nx = Nx_values(i);
    filename = sprintf('heat1dBcNNNx%d.m',Nx);
    if exist(filename, 'file') ~= 2
        error('File %s does not exist. Please execute heat1dNN to generate it.', filename);
    else
        run(filename);
        err = err(2:end-1); % interior points only
        maxErr = max(err);
        errors_NN(i) = maxErr;
        dx_values(i) = dx;
        fprintf('Nx=%d, dx=%e, max error=%e\n', Nx, dx, maxErr);
    end
end

% calculating the ratio and convergence rate
for m = 1:numTests-1
    ratio_DD = errors_DD(m) / errors_DD(m+1);
    rate_DD = log2(ratio_DD);
    ratio_NN = errors_NN(m) / errors_NN(m+1);
    rate_NN = log2(ratio_NN);
    fprintf('DD: Nx=%d to Nx=%d, Error Ratio=%4.2f, Convergence Rate=%4.2f\n', ...
        Nx_values(m), Nx_values(m+1), ratio_DD, rate_DD);
    fprintf('NN: Nx=%d to Nx=%d, Error Ratio=%4.2f, Convergence Rate=%4.2f\n', ...
        Nx_values(m), Nx_values(m+1), ratio_NN, rate_NN);
end

% Plotting the maximum error vs dx on a loglog scale
figure('Position', [100, 100, 800, 600]);
loglog(dx_values, errors_DD, '-bo', 'LineWidth', 2.0, 'MarkerSize', 6);
hold on;
loglog(dx_values, errors_NN, '-rs', 'LineWidth', 1.0, 'MarkerSize', 3);
grid on;
set(gca, 'FontSize', 12);
xlabel('Grid Spacing (dx)', 'FontSize', 14);
ylabel('Maximum Error', 'FontSize', 14);
title('Convergence Test for 1D Heat Equation', 'FontSize', 16);


% Creating a reference line based on the first DD error point
ref_error = errors_DD(1);  % Reference error at coarsest grid
ref_line = ref_error * (dx_values / dx_values(1)).^2;  % Second-order scaling
loglog(dx_values, ref_line, 'k--', 'LineWidth', 1.5);  % plots the reference line
legend('Dirichlet-Dirichlet BCs', 'Neumann-Neumann BCs', 'O(dx^2) Reference', 'Location', 'best');

% save the file as a png
saveas(gcf, 'convergence_test.png');