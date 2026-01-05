function spx_final = simulate_paths_vec(Y0, num_steps, numPaths, mu_Y, sigma_Y, dt, z)
    % 1. 计算所有 (num_steps) x numPaths 的对数收益率增量
    domega = sqrt(dt) * z;
    log_increments = (mu_Y - 0.5*sigma_Y^2) * dt + sigma_Y * domega;
    total_log_return = cumsum(log_increments, 1);
    spx_paths_all = Y0 .* exp(total_log_return);
     spx_final = spx_paths_all(end, :);

end
