% Fuzzy Systems 2018 - Group 2
% Kosmas Tsiakas 8255
% Car Control Ser03

%% CLEAR
clear all;
close all;

%% BEGIN
fprintf('\n *** begin %s ***\n\n',mfilename);

%% INITIALIZE
xo = 3.8; 
yo = 0.5;
u = 0.05; % constant speed
thetas = [0 45 -45]; % initial theta values
xd = 10; % desired position
yd = 3.2;

system = readfis('optimized-fuzzy-car-controller');
% Plot the membership functions
figure;
subplot(2,2,1)
plotmf(system, 'input',1);
title('Membership function of dv');

subplot(2,2,2)
plotmf(system, 'input',2);
title('Membership function of dh');

subplot(2,2,3)
plotmf(system, 'input',3);
title('Membership function of theta');

subplot(2,2,4)
plotmf(system, 'output',1);
title('Membership function of delta_theta');

for i = 1 : 1 : 3
    x = xo;
    y = yo;
    theta = thetas(i);
    
    lost = 0; % flag to check if the car is outside
    x_pos = []; % initialize to save later on
    y_pos = [];
    
    while (lost == 0)
        % Need to find the distances and then calculate the delta theta
        [dh, dv] =  get_distances(x, y);
        
        delta_theta = evalfis([dv dh theta], system);
        
        theta = theta + delta_theta;
        
        % New position of the car
        x = x + u * cosd(theta);
        y = y + u * sind(theta);
        
        % Check if the car is out of the map, so to exit
        if (x < 0) || (x>10) || (y <0) || (y > 4)
            lost = 1;
        end
        % Update the position
        x_pos = [x_pos; x];
        y_pos = [y_pos; y]; 
    end
    % Visualise the movement of the car and calculate the distance from the
    % desired position
    figure;
    error_x = xd -x;
    error_y = yd - y;
    
    obstacle_x = [5; 5; 6; 6; 7; 7; 10];
    obstacle_y = [0; 1; 1; 2; 2; 3; 3];
    
    title(['Starting angle: ', num2str(thetas(i)), ' | error in x-axis: ', num2str(error_x),' | error in y-axis: ', num2str(error_y)]);
    
    line(x_pos, y_pos, 'Color','blue');
    line(obstacle_x, obstacle_y, 'Color','red');
    % Mark the initial and desired points on the plot
    hold on;
    plot(xd, yd, '*');
    hold on;
    plot(xo, yo, '*');
end

%% FINISH
fprintf('\n *** %s has just finished ***\n\n',mfilename);
