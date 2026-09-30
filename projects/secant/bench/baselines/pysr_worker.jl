using Printf
using Statistics
using LoopVectorization
using SymbolicRegression
using SymbolicRegression.LossFunctionsModule: eval_loss

@inline secant_safe_div(x, y) = x * y / (y * y + 0.01f0)
@inline secant_safe_sqrt(x) = sqrt(abs(x) + 0.01f0)
@inline secant_safe_rsqrt(x) = inv(sqrt(abs(x) + 0.01f0))

const SECANT_FUNCTIONS = Dict{Symbol,Function}(
    :secant_safe_div => secant_safe_div,
    :secant_safe_sqrt => secant_safe_sqrt,
    :secant_safe_rsqrt => secant_safe_rsqrt,
)

const OPTIONS_STANDARD = Options(
    binary_operators=[+, -, *, /, min, max, secant_safe_div],
    unary_operators=[
        -,
        abs,
        sin,
        cos,
        exp2,
        log2,
        inv,
        tanh,
        secant_safe_sqrt,
        secant_safe_rsqrt,
    ],
    progress=false,
    turbo=false,
    bumper=false,
    should_optimize_constants=false,
)
const OPTIONS_TURBO = Options(
    binary_operators=[+, -, *, /, min, max, secant_safe_div],
    unary_operators=[
        -,
        abs,
        sin,
        cos,
        exp2,
        log2,
        inv,
        tanh,
        secant_safe_sqrt,
        secant_safe_rsqrt,
    ],
    progress=false,
    turbo=true,
    bumper=false,
    should_optimize_constants=false,
)

function value(args, name)
    index = findfirst(==(name), args)
    index === nothing && error("missing argument: $name")
    index == length(args) && error("missing value for argument: $name")
    return args[index + 1]
end

function parse_int_list(text)
    values = parse.(Int, split(text, ","))
    isempty(values) && error("integer list is empty")
    any(value -> value <= 0, values) && error("integer lists must be positive")
    return values
end

function resolve_secant_functions(expression)
    expression isa Expr || return expression
    arguments = Any[
        resolve_secant_functions(argument)
        for argument in expression.args
    ]
    if expression.head == :call && arguments[1] isa Symbol
        arguments[1] = get(SECANT_FUNCTIONS, arguments[1], arguments[1])
    end
    return Expr(expression.head, arguments...)
end

function hash32(value)
    value ⊻= value >> 16
    value *= UInt32(0x7feb352d)
    value ⊻= value >> 15
    value *= UInt32(0x846ca68b)
    value ⊻= value >> 16
    return value
end

function make_input(rows, seed, num_inputs)
    X = Matrix{Float32}(undef, num_inputs, rows)
    for column in 0:(num_inputs - 1)
        for row in 0:(rows - 1)
            bits = hash32(
                UInt32(column) * UInt32(0x9e3779b9) ⊻
                UInt32(row) * UInt32(0x85ebca6b) ⊻
                UInt32(seed) * UInt32(0xc2b2ae35) ⊻
                UInt32(0x51ed270b))
            X[column + 1, row + 1] =
                (Float32(bits & UInt32(0xffff)) / 32767.5f0 - 1.0f0) * 1.5f0
        end
    end
    return X
end

function make_trees(asts, expressions, variable_names)
    return [
        parse_expression(
            expressions[mod1(ast_idx, length(expressions))];
            operators=OPTIONS_STANDARD.operators,
            variable_names=variable_names,
        )
        for ast_idx in 1:asts
    ]
end

function materialize_serial(trees, dataset, options)
    checksum = Float32(0)
    @inbounds for tree in trees
        prediction, complete = eval_tree_array(tree, dataset.X, options)
        complete || error("eval_tree_array returned an incomplete result")
        checksum += prediction[1] + prediction[end]
    end
    return checksum
end

function materialize_threads(trees, dataset, options)
    checksums = zeros(Float32, length(trees))
    Threads.@threads :static for ast_idx in eachindex(trees)
        prediction, complete = eval_tree_array(trees[ast_idx], dataset.X, options)
        complete || error("eval_tree_array returned an incomplete result")
        checksums[ast_idx] = prediction[1] + prediction[end]
    end
    return sum(checksums)
end

function sse_serial(trees, dataset, options)
    checksum = Float32(0)
    @inbounds for tree in trees
        checksum += Float32(eval_loss(tree, dataset, options; regularization=false))
    end
    return checksum
end

function sse_threads(trees, dataset, options)
    checksums = zeros(Float32, length(trees))
    Threads.@threads :static for ast_idx in eachindex(trees)
        checksums[ast_idx] = Float32(
            eval_loss(trees[ast_idx], dataset, options; regularization=false))
    end
    return sum(checksums)
end

function time_case(batch, warmups, min_repeats, min_seconds)
    for _ in 1:warmups
        batch()
    end
    GC.gc()
    times = Float64[]
    total = 0.0
    checksum = 0.0
    while length(times) < min_repeats || total < min_seconds
        begin_ns = time_ns()
        checksum += batch()
        seconds = (time_ns() - begin_ns) * 1.0e-9
        push!(times, seconds)
        total += seconds
    end
    sort!(times)
    return minimum(times), median(times), length(times), checksum
end

function main()
    rows_values = parse_int_list(value(ARGS, "--rows"))
    ast_values = parse_int_list(value(ARGS, "--asts"))
    shapes = split(value(ARGS, "--shapes"), ",")
    evaluation_modes = split(value(ARGS, "--evaluation-modes"), ",")
    workers = parse(Int, value(ARGS, "--workers"))
    warmups = parse(Int, value(ARGS, "--warmups"))
    min_repeats = parse(Int, value(ARGS, "--min-repeats"))
    min_seconds = parse(Float64, value(ARGS, "--min-seconds"))
    seed = parse(Int, value(ARGS, "--seed"))
    corpus_name = value(ARGS, "--corpus-name")
    corpus_hash = value(ARGS, "--corpus-hash")
    corpus_profile = value(ARGS, "--corpus-profile")
    num_inputs = parse(Int, value(ARGS, "--num-inputs"))
    expressions = resolve_secant_functions.(
        Meta.parse.(readlines(value(ARGS, "--expression-file"))))
    target_expression = resolve_secant_functions(
        Meta.parse(strip(read(value(ARGS, "--target-file"), String))))
    variable_names = ["x$(idx)" for idx in 1:num_inputs]

    workers == Threads.nthreads() || error(
        "requested $workers workers but Julia started $(Threads.nthreads()) threads")
    all(shape -> shape in ("materialize", "sse"), shapes) || error("invalid shape")
    all(mode -> mode in ("standard", "turbo"), evaluation_modes) ||
        error("invalid evaluation mode")
    println("system,backend,shape,ast_mode,corpus,corpus_hash,seed,rows,asts,workers,execution_mode,best_seconds,median_seconds,asts_per_second,row_evals_per_second,repeats,checksum,notes")
    for rows in rows_values
        X = make_input(rows, seed, num_inputs)
        target_tree = parse_expression(
            target_expression;
            operators=OPTIONS_STANDARD.operators,
            variable_names=variable_names)
        target, complete = eval_tree_array(target_tree, X, OPTIONS_STANDARD)
        complete || error("target expression evaluation was incomplete")
        dataset = Dataset(X, target; variable_names=variable_names)
        for asts in ast_values
            trees = make_trees(asts, expressions, variable_names)
            for shape in shapes
                for evaluation_mode in evaluation_modes
                    options = evaluation_mode == "turbo" ? OPTIONS_TURBO : OPTIONS_STANDARD
                    batch = if shape == "materialize"
                        if workers == 1
                            () -> materialize_serial(trees, dataset, options)
                        else
                            () -> materialize_threads(trees, dataset, options)
                        end
                    else
                        if workers == 1
                            () -> sse_serial(trees, dataset, options)
                        else
                            () -> sse_threads(trees, dataset, options)
                        end
                    end
                    best, measured_median, repeats, checksum =
                        time_case(batch, warmups, min_repeats, min_seconds)
                    reduction_note =
                        shape == "sse" ? ";native_mean_squared_error" : ""
                    @printf(
                        "pysr_symbolicregression,cpu_symbolicregression_julia,%s,alu,%s,%s,%d,%d,%d,%d,%s_%s,%.9f,%.9f,%.3f,%.3f,%d,%.9g,low_level_julia_backend%s;turbo=%s;profile=%s\n",
                        shape,
                        corpus_name,
                        corpus_hash,
                        seed,
                        rows,
                        asts,
                        workers,
                        workers == 1 ? "serial" : "julia_threads",
                        evaluation_mode,
                        best,
                        measured_median,
                        asts / measured_median,
                        asts * rows / measured_median,
                        repeats,
                        checksum,
                        reduction_note,
                        evaluation_mode == "turbo" ? "true" : "false",
                        corpus_profile,
                    )
                end
            end
        end
    end
end

main()
