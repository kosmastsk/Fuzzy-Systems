% Fuzzy Systems 2018 - Group 1 - T9
% Kosmas Tsiakas 8255
% Linear PI controller

%% CLEAR
clear all;
close all;

%% BEGIN
fprintf('\n *** begin %s ***\n\n',mfilename);

%% INITIALIZE
% Gc(s)
numc = [1 5];
denc = [1 0];
gc = tf(numc, denc);

% Gp(s)
nump = [25];
denp = [1 10.1 1];
gp = tf(nump, denp);

% Open loop system
sys_open_loop = series(gc, gp);

% Create the root locus plot
figure;
rlocus(sys_open_loop)

% Closed loop system
K = 1;
sys_open_loop = K * sys_open_loop; % We have chosen K
sys_closed_loop = feedback(sys_open_loop, 1, -1);

figure;
step(sys_closed_loop);

info = stepinfo(sys_closed_loop);

if info.RiseTime > 0.6
    fprintf('Rise Time is : %d. Try another value.',info.RiseTime);
end
if info.Overshoot > 8
    fprintf('Overshoot is : %d. Try another value.',info.Overshoot);
end

fprintf('\n *** %s has finished ***\n\n',mfilename);