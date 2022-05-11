function [A, FixedPoint] = BASC(Mu, Nu, gpu)
        N = Mu;
        M = 2 * Nu;
        alphainit = 0.9;
        v = 2;
        vmax = 2^10;
        imax = 1e5;
        epsilon = 0.0000000001;
        Cs = randi(2, [N, M/2]);
        Cs = [Cs, -Cs];
        assert(isequal(size(Cs), [N, M]));
        if gpu
                Cs = gpuArray(Cs);
        end
        alpha = alphainit;
        something = 0
        while v < vmax
                FixedPoint = true;
                i = 0;
                while i < imax & FixedPoint
                        Fm = zeros([N, M/2]);
                        if gpu
                                gpuArray(Cs);
                        end
                        for m = 1:1:M/2
                                idx = ones([M,1]);
                                idx(m) = 0;
                                idx(m + Nu) = 0;
                                idx = boolean(idx);
                                sm = Cs(:, m);
                                sl = Cs(:, idx);
                                Fm(:, m) = sum(((sm - sl) ./ (vecnorm(sm - sl) .^v))')';
                                %fm = 0;
                                %for l = 1:1:M
                                %       if l ~=m & l ~= m + Nu
                                %               fm = fm + (Cs(:,m) - Cs(:, l)) / ((norm(Cs(:,m) - Cs(:,l)))^v);
                                %       end
                                %end
                                %Fm(:, m) = fm;
                        end
                        Cs(:, 1:1:M/2) = Cs(:, 1:1:M/2) + alpha * Fm;
                        Cs(:, M/2+1:1:M) = - Cs(:, 1:1:M/2);[v, alpha, mutual_coherence(Cs(:, 1:1:M/2)', 0)]
                        if sum(vecnorm(Cs(:, 1:1:M/2) - Fm) < epsilon) == M/2
                                FixedPoint = false;
                        end
                        i = (i + 1) * 1.2;
                end
                v = 2 * v
                alpha = alphainit / (v - 1)
        end
        A = Cs(:, 1:1:M/2);
        FixedPoint = ~FixedPoint;
