
function objective = obj_fun_iv_v3(parameter, option_matrix_1, option_matrix_2,...
    spx_dist_1, spx_dist_2, threshold, Z1, Z2, num_steps1, num_steps2)

% 提取优化参数 (在 CPU 上)
gamma0 = parameter(1); 
nu0 = parameter(2); 
theta_bar = parameter(3); 
zeta_theta = parameter(4); 
sigma_theta = parameter(5); 


T1 = option_matrix_1(1, 2) - option_matrix_1(1, 1);
T2 = option_matrix_2(1, 2) - option_matrix_2(1, 1); 
num_sim_1 = length(spx_dist_1); 
num_sim_2 = length(spx_dist_2); 
stock_p = option_matrix_1(1, 7); 
rf1 = option_matrix_1(1, 9); 
rf2 = option_matrix_2(1, 9); 
%传入GPU 创建为 'gpuArray' 
p_p_1 = gpuArray(ones(num_sim_1, 1) / num_sim_1);
p_p_2 = gpuArray(ones(num_sim_2, 1) / num_sim_2);

%% 蒙特卡洛模拟M_t

dt1 = 1/252; 
dt2 = 1/252; 

% 由于 Z1 是 gpuArray [14, 27]，所有这些操作将 *自动* 在 GPU 上执行 
a1 = exp(-zeta_theta * dt1); 
a2 = exp(-zeta_theta * dt2); 
coef1 = sigma_theta * sqrt((1 - exp(-2 * zeta_theta * dt1)) / (2 * zeta_theta)); 
coef2 = sigma_theta * sqrt((1 - exp(-2 * zeta_theta * dt2)) / (2 * zeta_theta)); 

% --- 标注 (GPU 修改): 'zeros'  必须显式创建为 'gpuArray' 
inc1 = coef1 * Z1; 
inc2 = coef2 * Z2;

theta_dev1 = filter(1, [1, -a1], inc1); %向量化递归
theta_dev2 = filter(1, [1, -a2], inc2); 
theta_t_1 = theta_bar + theta_dev1; 
theta_t_2 = theta_bar + theta_dev2; 

theta_0_vec_1 = gpuArray(ones(1, num_sim_1) * theta_bar);
theta_0_vec_2 = gpuArray(ones(1, num_sim_2) * theta_bar);

theta_prev1 = [theta_0_vec_1; theta_t_1(1:end-1, :)];
theta_prev2 = [theta_0_vec_2; theta_t_2(1:end-1, :)];
%累加法求M_t%
domega1 = sqrt(dt1) * Z1; 
domega2 = sqrt(dt2) * Z2; 
incM1 = theta_prev1.* domega1 - 0.5 * (theta_prev1.^ 2) * dt1; 
incM2 = theta_prev2.* domega2 - 0.5 * (theta_prev2.^ 2) * dt2; 
logM1 = cumsum(incM1, 1); 
logM2 = cumsum(incM2, 1); 
M_t_1 = exp(logM1); 
M_t_2 = exp(logM2); 

%% 计算定价核 m_1 和 m_2 
p = (gamma0 - 1) / (2 * gamma0 - 1);
% --- 标注 (GPU 修改):

M_final_1 = M_t_1(end, :)'; 
spx_dist_1_col = spx_dist_1'; 
u_1_val_1 = 1./ (spx_dist_1_col / stock_p); 
w_1_val_1 = (1 - nu0).* (M_final_1).^ p + nu0.* (M_final_1).^ (1 - p); 
m_1_unnorm = w_1_val_1.* u_1_val_1; 
expect_m_1 = m_1_unnorm' * p_p_1;  % GPU 矩阵乘法 
m_1 = m_1_unnorm.* (exp(-rf1 * T1 / 365) / expect_m_1);


M_final_2 = M_t_2(end, :)'; 
spx_dist_2_col = spx_dist_2'; 
u_2_val_1 = 1./ (spx_dist_2_col / stock_p); 
w_2_val_1 = (1 - nu0).* (M_final_2).^ p + nu0.* (M_final_2).^ (1 - p);
m_2_unnorm = w_2_val_1.* u_2_val_1; 
expect_m_2 = m_2_unnorm' * p_p_2;  % GPU 矩阵乘法 
m_2 = m_2_unnorm.* (exp(-rf2 * T2 / 365) / expect_m_2);


%% compute the theoretical options' prices 
[num_options_1, ~] = size(option_matrix_1);
real_p_1 = option_matrix_1(:, 11); 
real_iv_1 = option_matrix_1(:, 15); 

[num_options_2, ~] = size(option_matrix_2); 
real_p_2 = option_matrix_2(:, 11); 
real_iv_2 = option_matrix_2(:, 15); 

combined_weights_1 = m_1.* p_p_1; % [num_sim x 1] gpuArray
option_types_1 = option_matrix_1(:, 3);
strikes_1_cpu = option_matrix_1(:, 4);
call_idx_1 = (option_types_1 == 0);
put_idx_1 = (option_types_1 == 1);

% 将行权价移至 GPU
call_strikes_1 = gpuArray(strikes_1_cpu(call_idx_1)'); % [1 x K_call] gpuArray [16]
put_strikes_1 = gpuArray(strikes_1_cpu(put_idx_1)');   % [1 x K_put] gpuArray [16]

est_prices_call_1 = 0;
est_prices_put_1 = 0;

if any(call_idx_1)
    call_payoffs = max(spx_dist_1_col - call_strikes_1, 0); % [N x K_call] gpuArray

    est_prices_call_1 = combined_weights_1' * call_payoffs; % [1 x K_call] gpuArray
end
if any(put_idx_1)
    put_payoffs = max(put_strikes_1 - spx_dist_1_col, 0);  % [N x K_put] gpuArray
    est_prices_put_1 = combined_weights_1' * put_payoffs; % [1 x K_put] gpuArray
end

% 在 GPU 上组合结果
estimate_p_1_gpu = gpuArray(zeros(num_options_1, 1)); 
if any(call_idx_1)
    estimate_p_1_gpu(call_idx_1) = est_prices_call_1';
end
if any(put_idx_1)
    estimate_p_1_gpu(put_idx_1) = est_prices_put_1';
end

combined_weights_2 = m_2.* p_p_2;
option_types_2 = option_matrix_2(:, 3);
strikes_2_cpu = option_matrix_2(:, 4);
call_idx_2 = (option_types_2 == 0);
put_idx_2 = (option_types_2 == 1);

call_strikes_2 = gpuArray(strikes_2_cpu(call_idx_2)'); 
put_strikes_2 = gpuArray(strikes_2_cpu(put_idx_2)'); 
est_prices_call_2 = 0;
est_prices_put_2 = 0;

if any(call_idx_2)
    call_payoffs = max(spx_dist_2_col - call_strikes_2, 0);
    est_prices_call_2 = combined_weights_2' * call_payoffs;
end
if any(put_idx_2)
    put_payoffs = max(put_strikes_2 - spx_dist_2_col, 0);
    est_prices_put_2 = combined_weights_2' * put_payoffs;
end

estimate_p_2_gpu = gpuArray(zeros(num_options_2, 1)); 
if any(call_idx_2)
    estimate_p_2_gpu(call_idx_2) = est_prices_call_2';
end
if any(put_idx_2)
    estimate_p_2_gpu(put_idx_2) = est_prices_put_2';
end

estimate_p_1 = gather(estimate_p_1_gpu);
estimate_p_2 = gather(estimate_p_2_gpu);

calc_iv_vectorized = @(opt_mat, est_prices) compute_iv_batch(opt_mat, est_prices);

estimate_iv_1 = calc_iv_vectorized(option_matrix_1, estimate_p_1);
estimate_iv_2 = calc_iv_vectorized(option_matrix_2, estimate_p_2);


%disp('price Check1 :');
% disp(estimate_p_1(1));
% disp(real_p_1(1));
 %disp(estimate_p_2(1));
% disp(real_p_2(1));

%disp('IV Check1 (First 5):');
%disp([estimate_iv_1(1:min(5, end)), real_iv_1(1:min(5, end))]);

% disp('IV Check2 (First 5):');
%disp([estimate_iv_2(1:min(5, end)), real_iv_2(1:min(5, end))]);


%% compute the squared difference between the theoretical implied volatilities and real implied volatilities

objective = 0;
N=0;

for j=1:num_options_1 
    if real_p_1(j)>threshold
        if ~isnan(estimate_iv_1(j))
        objective = objective + (estimate_iv_1(j)-real_iv_1(j))^2/num_options_1; 
        N=N+1;
        end
    end
end

for j=1:num_options_2 
    if real_p_2(j)>threshold
        if ~isnan(estimate_iv_2(j))
        objective = objective + (estimate_iv_2(j)-real_iv_2(j))^2/num_options_2; 
        N=N+1;
        end
    end
end


if N > 0
    objective = objective / N;
else
    objective = 1e10; % 返回一个巨大的惩罚值
end

end