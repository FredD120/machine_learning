using HDF5
using GLMakie
using Scylla
using BenchmarkTools
import .Threads
import Serialization
import Random

const DATA_SIZE = 1_428_000    # max size zurichess = 1_428_000
const BATCH_SIZE = 4           # train in batches for efficiency
const K_VALUE = 4.0 / 500      # hyperparameter in sigmoid function
const LEARNING_RATE = 1f2      # small = slower convergence but no overshooting
const MAX_EPOCHS = 500         # final cutoff point for training

const PST_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
const PST_TABLE_SIZE = 64
const ORIGINAL_PST_FOLDER = "$(dirname(@__DIR__))/PST"
const UPDATED_PST_FOLDER = "$(dirname(@__DIR__))/PST_updated"

"retrieve piece square tables from file"
function get_pst(type, folder = ORIGINAL_PST_FOLDER)
    h5open(joinpath(folder, "$(type).h5"), "r") do fid
        MG::Vector{Float32} = read(fid["MidGame"])
        EG::Vector{Float32} = read(fid["EndGame"])
        return (MG, EG)
    end
end

function get_weights(folder = ORIGINAL_PST_FOLDER)
    (kingMG, kingEG) = get_pst(PST_TYPES[1], folder)
    (queenMG, queenEG) = get_pst(PST_TYPES[2], folder)
    (rookMG, rookEG) = get_pst(PST_TYPES[3], folder)
    (bishopMG, bishopEG) = get_pst(PST_TYPES[4], folder)
    (knightMG, knightEG) = get_pst(PST_TYPES[5], folder)
    (pawnMG, pawnEG) = get_pst(PST_TYPES[6], folder)

    weights = Float32[]

    for vec in [kingMG, queenMG, rookMG, bishopMG, knightMG, pawnMG,
                kingEG, queenEG, rookEG, bishopEG, knightEG, pawnEG]
        for v in vec
            push!(weights, v)
        end
    end
    return weights
end

"store flattened weights as piece square table files matching get_weights order"
function store_weights(weights::Vector{Float32}, folder = UPDATED_PST_FOLDER)
    expected_length = 2 * length(PST_TYPES) * PST_TABLE_SIZE
    length(weights) == expected_length || throw(ArgumentError("expected $expected_length weights, got $(length(weights))"))

    mkpath(folder)
    endgame_offset = length(PST_TYPES) * PST_TABLE_SIZE

    for (i, type) in enumerate(PST_TYPES)
        start_ind = (i - 1) * PST_TABLE_SIZE + 1
        end_ind = i * PST_TABLE_SIZE
        mg = copy(weights[start_ind:end_ind])
        eg = copy(weights[endgame_offset + start_ind:endgame_offset + end_ind])

        h5open(joinpath(folder, "$(type).h5"), "w") do fid
            fid["MidGame"] = mg
            fid["EndGame"] = eg
        end
    end
end

function pst_heatmap_matrix(table::Vector{Float32})
    length(table) == PST_TABLE_SIZE || throw(ArgumentError("expected $PST_TABLE_SIZE PST values, got $(length(table))"))
    return reverse(reshape(table, 8, 8), dims = 2)
end

"compare two PST folders and create one figure per piece with difference and original PST heatmaps"
function plot_weight_differences(
    original_folder = ORIGINAL_PST_FOLDER,
    new_folder = UPDATED_PST_FOLDER;
    save_folder = nothing)

    figures = Figure[]

    if save_folder !== nothing
        mkpath(save_folder)
    end

    for type in PST_TYPES
        (original_mg, original_eg) = get_pst(type, original_folder)
        (new_mg, new_eg) = get_pst(type, new_folder)
        mg_diff = new_mg .- original_mg
        eg_diff = new_eg .- original_eg
        scale = maximum(abs, vcat(mg_diff, eg_diff))
        scale = scale == 0 ? 1f0 : scale
        diff_colorrange = (-scale, scale)
        original_scale = maximum(abs, vcat(original_mg, original_eg))
        original_scale = original_scale == 0 ? 1f0 : original_scale
        original_colorrange = (0, original_scale)

        fig = Figure(size = (900, 820))
        Label(fig[1, 1:2], uppercasefirst(type), fontsize = 24)

        ax_mg_diff = Axis(fig[2, 1], title = "MidGame difference", aspect = DataAspect())
        ax_eg_diff = Axis(fig[2, 2], title = "EndGame difference", aspect = DataAspect())
        ax_mg_original = Axis(fig[3, 1], title = "Original MidGame", aspect = DataAspect())
        ax_eg_original = Axis(fig[3, 2], title = "Original EndGame", aspect = DataAspect())

        hm_diff = heatmap!(ax_mg_diff, pst_heatmap_matrix(mg_diff); colorrange = diff_colorrange, colormap = :RdBu)
        heatmap!(ax_eg_diff, pst_heatmap_matrix(eg_diff); colorrange = diff_colorrange, colormap = :RdBu)
        hm_original = heatmap!(ax_mg_original, pst_heatmap_matrix(original_mg); colorrange = original_colorrange, colormap = :RdBu)
        heatmap!(ax_eg_original, pst_heatmap_matrix(original_eg); colorrange = original_colorrange, colormap = :RdBu)

        Colorbar(fig[2, 3], hm_diff, label = "new - original")
        Colorbar(fig[3, 3], hm_original, label = "original")

        hidedecorations!(ax_mg_diff)
        hidedecorations!(ax_eg_diff)
        hidedecorations!(ax_mg_original)
        hidedecorations!(ax_eg_original)
        push!(figures, fig)

        if save_folder !== nothing
            save(joinpath(save_folder, "$(type)_weight_difference.png"), fig)
        end
    end

    return figures
end

struct Position
    features::Vector{Float32} # (white occupied positions) - (black occupied positions), for mid and endgame
    score::Float32            # score of position in actual game (win/draw/loss)
    occupied::Vector{Int}  # only visit feautures that are non-zero
    original_eval::Int16      # for testing evaluation dot product
end

function Position(fen::S, result::S) where {S <: AbstractString}
    board = BoardState(fen)
    mg_phase = Scylla.phase(Scylla.count_pieces(board))
    eg_phase = Scylla.endgame_phase(mg_phase)
    mg_phase = Float32(mg_phase / 256)
    eg_phase = Float32(eg_phase / 256)
    features = zeros(Float32, 768)

    for type in 1:6 #[KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN]
        for colour in [true, false] #[WHITE, BLACK]
            for pos in board.pieces[Scylla.long_index(colour) + type]
                sign = Scylla.sgn(colour)
                piece_ind = Scylla.side_index(colour, pos) + 1
                board_ind = (type - 1) * 64

                features[board_ind + piece_ind] += sign * mg_phase
                features[board_ind + piece_ind + 384] += sign * eg_phase
            end
        end
    end

    score = 0.0
    if occursin("1-0", result) # white wins
        score = 1.0
    elseif occursin("0-1", result) # black wins
        score = -1.0
    end
    
    occupied = Int[]
    values = Float32[]
    for (i, f) in enumerate(features)
        if f != 0
            push!(occupied, i)
            push!(values, f)
        end
    end

    return Position(values, score, occupied, Scylla.evaluate(board))
end

function get_positions(max_positions = DATA_SIZE)::Vector{Position}
    data = readlines(pwd() * "\\data\\zurichess.epd")
    position_count = min(max_positions, length(data))

    positions = Vector{Position}(undef, position_count)
    for i in 1:position_count
        arr = split(data[i], "c9")
        positions[i] = Position(arr[1], arr[2])
    end
    return positions
end 

function quality(pos::Position, pst)
    count = Float32(0)
    for (linear_index, jump_index) in enumerate(pos.occupied)
        count += pos.features[linear_index] * pst[jump_index]
    end
    return count
end

sigmoid(x, k) = 1 / (1 + exp(-k * x))

normalised_quality(pos::Position, weights, k) = 2 * sigmoid(quality(pos, weights), k) - 1

calculate_error(pos::Position, weights, k) = normalised_quality(pos, weights, k) - pos.score

function mean_squared_error(positions::Vector{Position}, weights::Vector{Float32}; k = K_VALUE)
    error = Float32(0.0)
    for position in positions
        diff = calculate_error(position, weights, k)
        error += diff * diff
    end
    return error / length(positions)
end

function create_feature_data(filename)
    positions = get_positions()
    Serialization.serialize(filename, positions)
end

function plot_error(feature_set, eval, BUCKET_COUNT = 100; k = K_VALUE)
    buckets = zeros(BUCKET_COUNT)
    for position in feature_set
        loss = calculate_error(position, eval, k)
        ind = floor(Int16, (BUCKET_COUNT - 1) * (loss + 2) / 4) + 1
        buckets[ind] += 1
    end
    xs = [-2 + 4 * i / BUCKET_COUNT for i in 1:BUCKET_COUNT]
    barplot(xs, buckets)
end

function plot_evaluation(feature_set::Vector{Position}, BUCKET_COUNT = 100)
    buckets = zeros(BUCKET_COUNT)
    for position in feature_set
        ind = floor(Int16, (BUCKET_COUNT - 1) * (1 + position.original_eval / 900) / 2)
        ind = min(BUCKET_COUNT, max(1, ind + 1))
        buckets[ind] += 1
    end
    barplot(buckets)
end

"loop over all weights and nudge them in the direction that minimises the loss"
function gradient_descent!(
    weights::Vector{Float32},
    positions::Vector{Position};
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    k = K_VALUE)

    Random.shuffle!(positions)
    gradient = zeros(Float32, length(weights))

    for start in 1:batch_size:length(positions)
        stop = min(start + batch_size - 1, length(positions))
        batch_len = 0
        fill!(gradient, 0f0)

        for i in start:stop
            position = positions[i]
            batch_len += 1
            y = position.score
            sigma_k = sigmoid(quality(position, weights), k)

            derivative = (2 * sigma_k - 1 - y) * (1 - sigma_k) * sigma_k

            for (linear_ind, jump_ind) in enumerate(position.occupied)
                gradient[jump_ind] += position.features[linear_ind] * derivative
            end
        end

        scale = Float32(4 * k * learning_rate / batch_len)
        for i in eachindex(weights, gradient)
            weights[i] -= scale * gradient[i]
        end
    end
end

function run_gradient_descent!(
    weights::Vector{Float32},
    positions::Vector{Position};
    max_epochs = MAX_EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    k = K_VALUE)

    count = 0
    losses = Float32[]
    while count < max_epochs
        count += 1
        gradient_descent!(weights, positions; batch_size, learning_rate, k)

        if count % 5 == 0
            loss = mean_squared_error(positions, weights; k)
            push!(losses, loss)
            println("Epoch ", count, " completed, loss = ", loss)
        end

    end
    return losses
end

function plot_loss(loss)
    fig = Figure()
    ax = Axis(fig[1, 1])
    ax.xlabel = "Epoch"
    ax.ylabel = "Loss"

    xs = [i for i in 1:length(loss)]
    GLMakie.lines!(ax, xs, loss)
    display(fig)
    return
end

function main()
    filename = "$(dirname(@__DIR__))/data/zurichess_serialize"
    #create_feature_data(filename)
    features = Serialization.deserialize(filename)
    weights = get_weights()

    @time losses = run_gradient_descent!(weights, features)
    #store_weights(weights)

    plot_loss(losses)

    #@btime c = evaluate_all($features, $weights)
    #@btime mean_squared_error($features, $weights)
    #plot_error(features, weights)
    #plot_evaluation(features)
end

function plot_results()
    plot_weight_differences(; save_folder = "$(dirname(@__DIR__))/PST_diff_plots")
    return
end

function parameter_scan(; filename = "$(dirname(@__DIR__))/data/zurichess_serialize")
    parameters = [()]
    final_loss = zeros(Float32, length(parameters))
    features = Serialization.deserialize(filename)
    
    Threads.@threads for i in eachindex(parameters)
        weights = get_weights()
        
        losses = run_gradient_descent!(weights, copy(features),
        k = parameters[i][1])
        final_loss[i] = losses[end]
    end
    println(final_loss)
    println(parameters)
end

main()
#plot_results()
#parameter_scan()
