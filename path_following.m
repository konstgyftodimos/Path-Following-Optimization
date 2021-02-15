%% Author: Konstantinos Gyftodimos
%% Title: Path following, MPC controller, Optimization

clear all;
clc;
import rockit.*

%% Define path

% Example path #1
xp = linspace(0,4,30);
yp = 5*exp(-0.2*xp);

% Example path #2
%xp = linspace(0,2,50);
%yp = -1.5*sqrt(4-xp.^2)+3;

% Example path #3
%xp = 3*sin(linspace(0,2*pi,50));
%yp = 8*linspace(1,2,50).^3;

path_length = length(xp);

%% Parameters

Nsim = 30;
Ncontrol = 10;
M = 1;

% Logging variables
x_history = zeros(Nsim+1,Ncontrol+1);
y_history = zeros(Nsim+1,Ncontrol+1);
yaw_history = zeros(Nsim+1,Ncontrol+1);
time_history = zeros(Nsim+1,Ncontrol+1);
speed_history = zeros(Nsim+1,Ncontrol+1);
steer_history = zeros(Nsim+1,Ncontrol+1);

comptime = zeros(Nsim+1,1);
error = zeros(Nsim+1,1);

%% Setting up OCP

ocp = Ocp('T',FreeTime(1.0));

% States
x = ocp.state();
y = ocp.state();
yaw = ocp.state();

% Controls
speed = ocp.control();
steer = ocp.control();

% ODEs
ocp.set_der(x,speed*cos(yaw));
ocp.set_der(y,speed*sin(yaw));
ocp.set_der(yaw,speed*tan(steer));

% Parameters for initial values
initial_x = ocp.parameter();
initial_y = ocp.parameter();
initial_yaw = ocp.parameter();

ocp.subject_to(ocp.at_t0(x) == initial_x);
ocp.subject_to(ocp.at_t0(y) == initial_y);
ocp.subject_to(ocp.at_t0(yaw) == initial_yaw);

% Initial guess
ocp.set_initial(x,0);
ocp.set_initial(y,0);
ocp.set_initial(yaw,0);
ocp.set_initial(speed,1);
ocp.set_initial(steer,0);

% Input constraints
max_speed = 10;
max_steer = pi/6;
ocp.subject_to(0 <= speed <= max_speed);
ocp.subject_to(-max_steer <= steer <= max_steer);

% Defining subset of points on the path and arrival point
% needed to set a penalty (due to MPC-like operation and to fulfill
% path following, not all points are used at once)
path_points = ocp.parameter(2, 'grid','control');
last_point = ocp.parameter(2);

% Objectives
wxy = 1;
wu = 1e-3;
wf = 10;
ocp.add_objective(ocp.sum(wxy*sumsqr([x;y]-path_points),'grid','control'))
ocp.add_objective(ocp.integral(wu*sumsqr([speed;steer])));
ocp.add_objective(wf*sumsqr(ocp.at_tf([x;y])-last_point))

% Solver setup
options = struct;
options.ipopt.print_level = 0;
options.expand = true;
options.print_time = false;
ocp.solver('ipopt',options);

ocp.method(MultipleShooting('N',Ncontrol,'M',M,'intg','rk'));

%% First solve of OCP

% Define N points on the path to travel
idx = 1;

ocp.set_value(path_points,[xp(1:Ncontrol);yp(1:Ncontrol)]);
ocp.set_value(last_point,[xp(Ncontrol);yp(Ncontrol)]);

% Starting position is first point on the path
current_x = xp(1);
current_y = yp(1);
current_yaw = 0;
ocp.set_value(initial_x,current_x);
ocp.set_value(initial_y,current_y);
ocp.set_value(initial_yaw, current_yaw);

% Solve
sol = ocp.solve();

% Get system dynamics for iterating
dynamics = ocp.discrete_system;

% Storing results
[t_sol,x_sol] = sol.sample(x,'grid','control');
[t_sol,y_sol] = sol.sample(y,'grid','control');
[t_sol,yaw_sol] = sol.sample(yaw,'grid','control');
[t_sol,speed_sol] = sol.sample(speed,'grid','control');
[t_sol,steer_sol] = sol.sample(steer,'grid','control');

x_history(1,:) = x_sol;
y_history(1,:) = y_sol;
yaw_history(1,:) = yaw_sol;
speed_history(1,:) = speed_sol;
steer_history(1,:) = steer_sol;
time_history(1,:) = t_sol;

error(1) = sol.value(ocp.objective);

statistics = struct(sol.parent.sol.opti_wrapper.stats());
comptime(1) = statistics.t_wall_callback_fun+statistics.t_wall_nlp_f+...
                      statistics.t_wall_nlp_g+statistics.t_wall_nlp_grad+...
                      statistics.t_wall_nlp_grad_f+...
                      statistics.t_wall_nlp_hess_l+statistics.t_wall_nlp_jac_g;

%% MPC loop

for i = 1:Nsim
    disp([num2str(i+1) '/' num2str(Nsim+1)])
    
    dt = t_sol(2)-t_sol(1);
    states = getfield(dynamics('x0',[current_x; current_y; current_yaw],'u',[speed_sol(1); steer_sol(1)],'T',dt),'xf');
    current_x = full(states(1));
    current_y = full(states(2));
    current_yaw = full(states(3));

    % Setting new starting state values
    ocp.set_value(initial_x,current_x);
    ocp.set_value(initial_y,current_y);
    ocp.set_value(initial_yaw,current_yaw);
    
    % Getting closest point on path to current position,
    % to decide on the start of the next set of objective points
    % (only taking in account after 'idx' to ensure we are not going
    % backwards
    distance = (xp(idx:end)-current_x).^2 + (yp(idx:end)-current_y).^2;
    idx = idx + find(distance == min(distance),1);
    if idx > length(xp)
        idx = length(xp); 
    end
    
    % in case of overflow (close to the end of the path) we append the
    % finishing point a sufficient number of times at the end
    if idx+Ncontrol-1 > length(xp)
        ocp.set_value(path_points,[xp(idx:end),xp(end)*ones(1,idx+Ncontrol-1-length(xp));...
                                     yp(idx:end),yp(end)*ones(1,idx+Ncontrol-1-length(yp))]);
        ocp.set_value(last_point,[xp(end);yp(end)]);
    else
        ocp.set_value(path_points,[xp(idx:idx+Ncontrol-1);yp(idx:idx+Ncontrol-1)]);
        ocp.set_value(last_point,[xp(idx+Ncontrol-1);yp(idx+Ncontrol-1)]);
    end
    % Solving, logging
    sol = ocp.solve();
    
    [t_sol,x_sol] = sol.sample(x,'grid','control');
    [t_sol,y_sol] = sol.sample(y,'grid','control');
    [t_sol,yaw_sol] = sol.sample(yaw,'grid','control');
    [t_sol,speed_sol] = sol.sample(speed,'grid','control');
    [t_sol,steer_sol] = sol.sample(steer,'grid','control');

    x_history(i+1,:) = x_sol;
    y_history(i+1,:) = y_sol;
    yaw_history(i+1,:) = yaw_sol;
    speed_history(i+1,:) = speed_sol;
    steer_history(i+1,:) = steer_sol;
    time_history(i+1,:) = t_sol;

    error(i+1) = sol.value(ocp.objective);
    statistics = struct(sol.parent.sol.opti_wrapper.stats());
    comptime(i+1) = statistics.t_wall_callback_fun+statistics.t_wall_nlp_f+...
                          statistics.t_wall_nlp_g+statistics.t_wall_nlp_grad+...
                          statistics.t_wall_nlp_grad_f+...
                          statistics.t_wall_nlp_hess_l+statistics.t_wall_nlp_jac_g;
    
    
    ocp.set_initial(x,x_sol);
    ocp.set_initial(y,y_sol);
    ocp.set_initial(yaw,yaw_sol);
    ocp.set_initial(speed,speed_sol);
    ocp.set_initial(steer,steer_sol);
end

%% Plotting results

close all;

dt = time_history(:,2) - time_history(:,1);
time = cumsum([0; dt(1:end-1)]);

% End index for plotting to remove the part when the controller
% is only stabilizing; only for visuality, as then the error is low,
% position is stable, but the time horizon is greater
end_idx = find(error < 1e-5, 1);
% next line is (un)commented based on graph visibility:
% commented for path #2 in our case
end_idx = end_idx - 1;
% after end_idx a long time passes for the next iteration, afterwards,
% position, error and control signals are constant, this cutting only
% disrupts the graph

figure(1)
subplot(211),
plot(time(1:end_idx),error(1:end_idx),'-o')
xlabel('Time [s]')
ylabel('Error')
subplot(212)
semilogy(time(1:end_idx),dt(1:end_idx),'bo',time(1:end_idx),comptime(1:end_idx),'ro')
xlabel('Time [s]')
legend({'Sample time [s]'; 'Computation time [s]'})

figure(2)
subplot(211)
ylim([0, 1.2*max_speed])
plot(time(1:end_idx),speed_history((1:end_idx),1),'-o')
xlabel('Time [s]')
ylabel('Speed')
subplot(212)
ylim(1.2*[-max_steer, max_steer])
plot(time(1:end_idx),steer_history((1:end_idx),1),'-o')
xlabel('Time [s]')
ylabel('Steering angle [rad]')

figure(3)
plot(xp,yp,'r.-')
xlim(1.2*[min(xp),max(xp)])
ylim(1.2*[min(yp),max(yp)])
hold on
xlabel('x')
ylabel('y')
plot(x_history(:,1),y_history(:,1),'bo')