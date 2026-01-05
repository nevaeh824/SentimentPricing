%从1996年1月开始 d=61


parallel.gpu.enableCUDAForwardCompatibility(true);
reset(gpuDevice);
clear;
clc;
tic; 
load('pw.mat');

N = 244;
threshold = 0;
N_days_per_year = 252;
numPaths = 200000;  % 蒙特卡洛路径数 
results = [];

% 优化参数范围设置
gamma0_range = 1.2;
nu0_range = 0.5;
theta_bar_range =0; 
zeta_theta_range = 1.2;  
sigma_theta_range = 1.2;

lb = [1.001,  0.0, -20,  0.001, 0.001];
ub = [20, 1.0,  20,  10, 10];

% 定义 ARIMA-GARCH 模型结构 (放在循环外以避免重复创建对象，提高效率)
% 均值方程: AR(1), 方差方程: GARCH(1,1)
Mdl_Spec = arima('ARLags', 1, 'Variance', garch(1,1)); 

for d =2
    
    if isnan(t_1(d)) || isnan(t_2(d)) || isnan(t_3(d))
        fprintf('d = %d: 索引为 NaN，跳过此月份。\n', d)
        continue; 
    end
    fprintf('Month d = %d: ', d)
    
    % --- 1. 获取日期与价格数据 ---
    date = options(t_1(d), 1);
    j = 1;
    while (spx(j,1) < date)
        j = j + 1;
    end 
    
    % 获取期权数据
    option_matrix_1 = options(t_1(d):t_2(d)-1, :);
    option_matrix_2 = options(t_2(d):t_3(d)-1, :);
    Y0 = spx(j, 2);  % 当前收盘价
    
    % --- 2. 提前计算时间步长 (为了确定 GARCH 预测的长度) ---
    T_1 = floor((options(t_1(d), 2) - options(t_1(d), 1)) / 7 * 5);
    dt1 = 1 / 252;
    num_steps1 = T_1;
    
    T_2 = floor((options(t_2(d), 2) - options(t_2(d), 1)) / 7 * 5);
    dt2 = 1 / 252;
    num_steps2 = T_2;
    
    forecast_horizon = max(num_steps1, num_steps2); % 预测视窗取较长的期权期限
    if forecast_horizon < 1
        forecast_horizon = 1; % 防止数据错误导致0步
    end

    % --- 3. ARIMA-GARCH 估计与预测 ---
    price_window = spx(j-1250:j, 2);
    log_returns = log(price_window(2:end) ./ price_window(1:end-1));
    
    fprintf('正在拟合 ARIMA-GARCH... ');
        % 估计模型参数
        EstMdl = estimate(Mdl_Spec, log_returns, 'Display', 'off');        
        % 预测未来 forecast_horizon 天的均值和方差
        [Y_F, ~, V_F] = forecast(EstMdl, forecast_horizon, 'Y0', log_returns);
        
        % 将 GARCH 预测值转换为 GBM 参数        
        % V_F 是每日方差预测序列
        avg_daily_variance = mean(V_F);
        sigma_Y = sqrt(avg_daily_variance * N_days_per_year);
        
        avg_daily_log_ret = mean(Y_F);  
        % 关系: mu_GBM * dt = E[log_ret] + 0.5 * sigma^2 * dt
        % 因此: mu_GBM = (E[log_ret] / dt) + 0.5 * sigma^2
        mu_Y = (avg_daily_log_ret * N_days_per_year) + (0.5 * sigma_Y^2);
        
        fprintf('完成. Sigma: %.4f, Mu: %.4f\n,d: %.4f',  sigma_Y, mu_Y,d);
        

    % --- 4. 进行蒙特卡洛模拟 ---
    Z = randn(max(num_steps2, num_steps1), numPaths);
    Z1 = Z(1:num_steps1, :); 
    Z2 = Z(1:num_steps2, :); 
    
    spx_dist_1 = simulate_paths_vec(Y0, num_steps1, numPaths, mu_Y, sigma_Y, dt1, Z1); 
    spx_dist_2 = simulate_paths_vec(Y0, num_steps2, numPaths, mu_Y, sigma_Y, dt2, Z2); 
    
    % --- 5. 上传 GPU ---
    % fprintf('正在将数据传输到 GPU...\n');
    try
        Z1_gpu = gpuArray(Z1);
        Z2_gpu = gpuArray(Z2);
        spx_dist_1_gpu = gpuArray(spx_dist_1);
        spx_dist_2_gpu = gpuArray(spx_dist_2);
    catch ME
        fprintf('GPU 初始化失败。\n');
        return; 
    end
    
    % --- 6. GMM 估计 (fmincon) ---
    condict = optimoptions('fmincon', ...
        'Display', 'off', ...          
        'TolFun', 1e-7, ...             
        'TolX', 1e-7, ...               
        'StepTolerance', 1e-7, ...
        'MaxFunctionEvaluations', 10000, ... 
        'MaxIterations', 500, ...
        'FiniteDifferenceType', 'central'); 
    
    x0 = [gamma0_range, nu0_range, theta_bar_range, zeta_theta_range, sigma_theta_range];
    
    [x_opt, fval] = fmincon(@(x) obj_fun_iv_v3(x, option_matrix_1, option_matrix_2,...
        spx_dist_1_gpu, spx_dist_2_gpu, threshold, Z1_gpu, Z2_gpu, num_steps1, num_steps2), ...
        x0, [], [], [], [], lb, ub, [], condict);
        
    results = [results; d, x0, x_opt, fval];
    
end
toc;
column_names = {'d', ...
                'gamma0_init', 'nu0_init', 'theta_bar_init', 'zeta_theta_init', 'sigma_theta_init', ...
                'gamma0_opt', 'nu0_opt', 'theta_bar_opt', 'zeta_theta_opt', 'sigma_theta_opt', ...
                'fval'};

% 将结果矩阵转换为 Table
results_table = array2table(results, 'VariableNames', column_names);

%  保存为 CSV 文件
filename = 'GMM_Optimization_Results.csv';
writetable(results_table, filename);

fprintf('\n✅ 结果已成功保存到文件: %s\n', filename);