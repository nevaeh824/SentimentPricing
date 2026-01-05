%数据可视化

%% 1. 数据读取与预处理
clear; clc;


% 读取 CSV 文件
filename = 'GMM_Optimization_Results.csv';
try
    raw_data = readmatrix(filename);
catch
    error('未找到文件 %s，请确认文件路径。', filename);
end

% 参数设定
%       Gamma, Nu, Theta, Zeta, Sigma_Theta
lb_vec = [1,    0.0, -10.0, 0 ,0];
ub_vec = [15.0, 1.0,  10.0, 5.0, 5.0];
param_names = {'Risk Aversion (\gamma_0)', 'Mixture Weight (\nu_0)', ...
               'Sentiment Mean (\theta)', 'Mean Reversion (\zeta_\theta)', ...
               'Sentiment Volatility (\sigma_\theta)'};
col_indices = [7, 8, 9, 10, 11]; % CSV中对应的列号

% --- 颜色设定 (5种鲜艳颜色) ---
line_colors = lines(5); 
line_colors(3, :) = [0.9290 0.6940 0.1250]; % 微调黄色使其更清晰

% --- 时间轴处理 ---
% 1. 生成绘图用的时间轴 (只包含有数据的月份，保证连线不断)
d_existing = raw_data(:, 1);
start_date = datetime(1996, 1, 2);
dates_existing = start_date + calmonths(d_existing - 1);

% 2. 生成完整时间轴 (仅用于计算X轴范围和检查缺失值)
d_full = (1:244)'; 
dates_full = start_date + calmonths(d_full - 1);

%% 2. 绘图设置
figure('Name', 'Parameter Estimates (Continuous Lines)', 'Color', 'w', 'Position', [100, 50, 1200, 1000]);
t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Estimated Parameters Time Series (1996-2016)', 'FontSize', 16, 'FontWeight', 'bold');

% 辅助线颜色
mean_color = [0.4, 0.4, 0.4]; % 灰色
bound_color = [0.8, 0, 0];    % 深红色

%% 3. 循环绘制 5 个参数图
for i = 1:5
    col_idx = col_indices(i);
    y_vals = raw_data(:, col_idx); % 直接使用原始数据，不含 NaN
    
    curr_color = line_colors(i, :);
    
    y_mean = mean(y_vals);
    y_lb = lb_vec(i);
    y_ub = ub_vec(i);
    
    nexttile;
    hold on;
    
    % 1. 绘制参数线 (连续线条)
    plot(dates_existing, y_vals, '.-', 'MarkerSize', 10, 'LineWidth', 1.5, 'Color', curr_color);
    
    % 2. 绘制平均值线
    yline(y_mean, '-', sprintf('Mean: %.2f', y_mean), ...
        'Color', mean_color, 'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left', 'FontSize', 9);
    
    % 3. 绘制上下界线
    yline(y_lb, '--', sprintf('LB: %.1f', y_lb), 'Color', bound_color, 'LabelHorizontalAlignment', 'right', 'Alpha', 0.5);
    yline(y_ub, '--', sprintf('UB: %.1f', y_ub), 'Color', bound_color, 'LabelHorizontalAlignment', 'right', 'Alpha', 0.5);
    
    % 特殊处理：给 Theta 图加 0 线
    if i == 3
        yline(0, '-', 'Color', 'k', 'Alpha', 0.3); 
    end

    % 4. 格式调整
    title(param_names{i}, 'FontSize', 12, 'FontWeight', 'bold', 'Color', curr_color);
    grid on; box on;
    xlim([min(dates_full), max(dates_full)]); % X轴范围依然覆盖全时段
    
    % 动态调整 Y 轴
    curr_ylim = ylim;
    new_ylim = [min(curr_ylim(1), y_lb - abs(y_lb)*0.1), max(curr_ylim(2), y_ub + abs(y_ub)*0.1)];
    if i == 2 
        ylim([-0.1, 1.1]);
    else
        ylim(new_ylim); 
    end
    
    hold off;
end

% --- 第 6 张图：拟合误差 ---
nexttile;
fval_vals = raw_data(:, 12);
semilogy(dates_existing, fval_vals, '.-', 'Color', [0 0 0]); % 纯黑，连续线
title('Calibration Error (MSE)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xlim([min(dates_full), max(dates_full)]);

% 全局 X 轴格式
linkaxes(findall(gcf,'type','axes'), 'x');
xtickformat('yyyy-MM');
xlabel(t, 'Date', 'FontSize', 12);

%% 4. 缺失月份总结 (保持逻辑不变)
fprintf('--------------------------------------------------\n');
fprintf('数据缺失月份统计 (Total Skipped: %d)\n', length(d_full) - length(d_existing));
missing_d = setdiff(d_full, d_existing);
for k = 1:length(missing_d)
    d_val = missing_d(k);
    missing_date = start_date + calmonths(d_val - 1);
    fprintf('d = %d: %s\n', d_val, datestr(missing_date, 'yyyy-mm'));
end
fprintf('--------------------------------------------------\n');