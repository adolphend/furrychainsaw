function [A, FixedPoint] = BASC(Mu, Nu, gpu)
        N = Mu;
        M = 2 * Nu;
        alphainit = 0.9;
        v = 2;
        vmax = 2^10;
        imax = 1e5;
        epsilon = 0.0000000001;
        Cs = rand([N, M/2]);
        Cs = [Cs, -Cs];
        assert(isequal(size(Cs), [N, M]));
        if gpu
                Cs = gpuArray(Cs);
        end
        zero = zeros([Mu, Nu]);
        if gpu
                gpuArray(zero);
        end
        alpha = alphainit;
        Cs = Cs ./ vecnorm(Cs);
        while v < vmax
                FixedPoint = true;
                i = 0;
                while i < imax & FixedPoint
                        Fm = zero;
                        for m = 1:1:M/2
                                idx = ones([M,1]);
                                idx(m) = 0;
                                idx(m + Nu) = 0;
                                idx = boolean(idx);
                                sm = Cs(:, m);
                                sl = Cs(:, idx);
                                Fm(:, m) = sum(((sm - sl) ./ (vecnorm(sm - sl) .^v))')';
                        end
                        Fm = Fm ./ vecnorm(Fm);
                        Cs(:, 1:1:M/2) = Cs(:, 1:1:M/2) + alpha * Fm;
                        Cs(:, 1:1:M/2) = Cs(:, 1:1:M/2) ./ vecnorm(Cs(:, 1:1:M/2));
                        Cs(:, M/2+1:1:M) = - Cs(:, 1:1:M/2);
                        if sum(vecnorm(Cs(:, 1:1:M/2) - Fm) < epsilon) == M/2
                                FixedPoint = false
                        end
                        i = (i + 1);
                end
                v = 2 * v;
                alpha = alphainit / (v - 1);
        end
