function iv_est = compute_iv_batch(opt_matrix, prices)
    % 提取列向量
    S = opt_matrix(:, 7);
    K = opt_matrix(:, 4);
    r = opt_matrix(:, 9);
    T = (opt_matrix(:, 2) - opt_matrix(:, 1)) / 365;
    q = opt_matrix(:, 8); % Yield
    is_call = (opt_matrix(:, 3) == 0);
     % 初始化输出 IV
    iv_est = NaN(size(prices)); 
    
    
    % 1. 处理所有的 Call
    idx_c = find(is_call);
    if ~isempty(idx_c)
        try
            % 一次性传入向量，速度极快
            iv_est(idx_c) = blsimpv(S(idx_c), K(idx_c), r(idx_c), T(idx_c), ...
                                     prices(idx_c), ...
                                    'Yield', q(idx_c), ...
                                    'Class', {'Call'}); % Class 支持单字符串自动广播
        catch
            % 如果批量失败(极罕见)，回退到逐个计算或保持 NaN
            % 此处为了效率，若失败通常意味着参数极度异常
        end
    end
    
    % 2. 处理所有的 Put
    idx_p = find(~is_call);
    if ~isempty(idx_p)
        try
            iv_est(idx_p) = blsimpv(S(idx_p), K(idx_p), r(idx_p), T(idx_p), ...
                                    safe_prices(idx_p), ...
                                    'Yield', q(idx_p), ...
                                    'Class', {'Put'});
        catch
        end
    end
end