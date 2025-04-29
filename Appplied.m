%% Energy Trading with ADMM
clear; clc;

%% Parameters
num_agents = 5;
N = 6;  % Time horizon

c_grid = 0.25;
c_battery = 0.05;
c_trade = 0.18;

battery_capacity = 2;
p_batt_max = 1;
p_grid_max = 2;      % Relaxed grid capacity
p_trade_max = 4;     % Relaxed trade capacity

max_iters = 50;
rho = 1;
tol = 1e-3;

%% Demand and Renewable Profiles
demand = rand(num_agents, N) * 2 + 2;   % Moderate demand 2-4 kW
renewable = zeros(num_agents, N);
renewable(1:2, :) = rand(2, N) * 5 + 3;
renewable(3:5, :) = rand(3, N) * 1;

%% Initialize 
for i = 1:num_agents
    agents(i).x0 = rand * battery_capacity;
    agents(i).P_grid = zeros(N,1);
    agents(i).P_batt = zeros(N,1);
    agents(i).P_trade = zeros(N,1);
    agents(i).lambda = zeros(N,1);
    agents(i).neighbor = mod(i, num_agents) + 1;
end

%% ADMM 
for iter = 1:max_iters
    for i = 1:num_agents
        neigh = agents(i).neighbor;

        P_trade_neigh = agents(neigh).P_trade;
        lambda = agents(i).lambda;

        f = [c_grid*ones(N,1); c_battery*ones(N,1); -c_trade*ones(N,1)] + ...
            [zeros(2*N,1); rho*(P_trade_neigh + lambda)];

        H = blkdiag(zeros(2*N), rho*eye(N));

        Aeq = [eye(N), eye(N), eye(N)];
        beq = demand(i,:)' - renewable(i,:)';

        lb = [zeros(N,1); -p_batt_max*ones(N,1); -p_trade_max*ones(N,1)];
        ub = [p_grid_max*ones(N,1); p_batt_max*ones(N,1); p_trade_max*ones(N,1)];

        options = optimoptions('quadprog','Display','off');
        [sol, ~, exitflag] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

        if exitflag > 0
            agents(i).P_grid = sol(1:N);
            agents(i).P_batt = sol(N+1:2*N);
            agents(i).P_trade = sol(2*N+1:end);
        else
            warning('Agent %d: QP failed at iteration %d', i, iter);
            agents(i).P_grid = zeros(N,1);
            agents(i).P_batt = zeros(N,1);
            agents(i).P_trade = zeros(N,1);
        end
    end

    converged = true;
    for i = 1:num_agents
        neigh = agents(i).neighbor;
        trade_gap = agents(i).P_trade + agents(neigh).P_trade;
        agents(i).lambda = agents(i).lambda + trade_gap;

        if norm(trade_gap) > tol
            converged = false;
        end
    end

    if converged
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

total_cost = zeros(num_agents,1);
for i = 1:num_agents
    grid_cost = c_grid * sum(agents(i).P_grid);
    batt_cost = c_battery * sum(abs(agents(i).P_batt));
    trade_profit = c_trade * sum(agents(i).P_trade);

    total_cost(i) = grid_cost + batt_cost - trade_profit;
    fprintf('Agent %d: Total Cost = $%.2f, Energy Traded = %.2f kWh\n', i, total_cost(i), sum(agents(i).P_trade));
end

%% Plot Energy Trading
figure;
bar_data = cell2mat(arrayfun(@(a) a.P_trade', agents, 'UniformOutput', false))';
bar(bar_data);
title('Energy Traded Over Time');
xlabel('Time Step');
ylabel('Energy Traded (kWh)');
legend_entries = arrayfun(@(i) sprintf('Agent %d',i), 1:num_agents, 'UniformOutput', false);
legend(legend_entries(1:size(bar_data,2)));
grid on;

%% Plot Total Costs
figure;
bar(total_cost);
title('Total Cost per Agent');
xlabel('Agent');
ylabel('Cost ($)');
grid on;

