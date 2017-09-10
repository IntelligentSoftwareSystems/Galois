%
% matlab_chol - Cholesky factorization using edgelist I/O
%

function matlab_chol
fn = getenv('CHOL_INPUT');              % Pass from shell via environment
% Load sparse matrix
F = load(fn, '-ascii');
% Change from zero-based indexes to one-based indexes
F(:, 1:2) = F(:, 1:2) + 1;
% Convert to a MATLAB sparse matrix
A = spconvert(F);
% Make symmetric
A = tril(A, -1) + triu(A);
% Do Cholesky factorization
tic
L = chol(A, 'lower');
toc
% Determine output filename
fn = getenv('CHOL_OUTPUT');
if length(fn) == 0
    fn = 'matlabedges.txt';
end
% Output sparse matrix
[i, j, x] = find(L);
i = i - 1;
j = j - 1;
DBL_DIG = 16;
fh = fopen(fn, 'w');
for k = 1:length(x)
    fprintf(fh, '%u %u %.19e\n', j(k), i(k), x(k));
end
status = fclose(fh);
assert(status == 0)