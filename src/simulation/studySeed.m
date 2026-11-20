function seed = studySeed(namespace, subscripts, sizes)
    % STUDYSEED Collision-free deterministic seed for one study arm.
    %
    % seed = studySeed(namespace, subscripts, sizes) maps one-based subscripts through a mixed-radix linear index and places the result in a namespace of its own.
    % The namespaces are far apart, so adding cells to one arm can never silently collide with another arm.
    % The seed is a pure function of the subscripts and the declared grid sizes: it does not depend on which cells a particular run executes, so a reduced run reproduces the replicates of the full run bit for bit.
    %
    % Inputs:
    %   namespace  - namespace name (string/char), one of
    %                  recovery  [1600000000, 1699999999]  parameter recovery
    %                  factorial [1700000000, 1799999999]  the correlation factorial
    %                  surrogate [1800000000, 1899999999]  the Gaussian surrogates
    %                  check     [1900000000, 1999999999]  the simulator check of Phase 7
    %   subscripts - 1 x k one-based subscripts, each <= its size.
    %   sizes      - 1 x k declared grid sizes, the mixed-radix radices.
    %
    % Output:
    %   seed - scalar double seed, inside the namespace and below 2^32, the documented range of RandStream('twister', 'Seed', .).

    arguments
        namespace (1, 1) string
        subscripts (1, :) double {mustBeInteger, mustBePositive}
        sizes (1, :) double {mustBeInteger, mustBePositive}
    end

    assert(numel(subscripts) == numel(sizes), 'studySeed:DimensionMismatch', ...
           'subscripts and sizes must have the same length; got %d and %d.', ...
           numel(subscripts), numel(sizes));
    assert(all(subscripts <= sizes), 'studySeed:IndexOutOfRange', ...
           'Every subscript must be no larger than its size; got %s against %s.', ...
           mat2str(subscripts), mat2str(sizes));

    namespaces = struct( ...
                        'recovery',  [1600000000, 1699999999], ...
                        'factorial', [1700000000, 1799999999], ...
                        'surrogate', [1800000000, 1899999999], ...
                        'check',     [1900000000, 1999999999]);
    key = char(namespace);
    assert(isfield(namespaces, key), 'studySeed:UnknownNamespace', ...
           'Unknown study seed namespace "%s"; expected one of %s.', ...
           key, strjoin(fieldnames(namespaces)', ', '));

    linearIndex = uint64(0);
    for k = 1:numel(sizes)
        linearIndex = linearIndex * uint64(sizes(k)) + uint64(subscripts(k) - 1);
    end
    range = namespaces.(key);
    seed64 = uint64(range(1)) + linearIndex + 1;
    assert(seed64 <= uint64(range(2)), 'studySeed:NamespaceExhausted', ...
           'Namespace "%s" needs more seeds (%s) than its reserved range holds.', ...
           key, num2str(double(seed64)));
    assert(seed64 < uint64(2)^32, 'studySeed:SeedAboveStreamLimit', ...
           'Seed %s is at or above 2^32, which RandStream cannot accept.', ...
           num2str(double(seed64)));
    seed = double(seed64);
end
