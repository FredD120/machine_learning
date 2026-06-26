using HDF5
using GLMakie
using Scylla
using BenchmarkTools
import .Threads
import Serialization
import Random

const DATA_SIZE = 1_428_000       # max size zurichess = 1_428_000
const BATCH_SIZE = 8            # train in batches for efficiency
const K_VALUE = 3.0 / 500         # hyperparameter in sigmoid function
const LEARNING_RATE = 1f2

"retrieve piece square tables from file"
function get_pst(type)
    h5open("$(dirname(@__DIR__))/PST/$(type).h5", "r") do fid
        MG::Vector{Float32} = read(fid["MidGame"])
        EG::Vector{Float32} = read(fid["EndGame"])
        return (MG, EG)
    end
end

function get_evaluation()
    (kingMG, kingEG) = get_pst("king")
    (queenMG, queenEG) = get_pst("queen")
    (rookMG, rookEG) = get_pst("rook")
    (bishopMG, bishopEG) = get_pst("bishop")
    (knightMG, knightEG) = get_pst("knight")
    (pawnMG, pawnEG) = get_pst("pawn")

    eval = Float32[]

    for vec in [kingMG, queenMG, rookMG, bishopMG, knightMG, pawnMG,
                kingEG, queenEG, rookEG, bishopEG, knightEG, pawnEG]
        for v in vec
            push!(eval, v)
        end
    end
    return eval
end

struct Position
    features::Vector{Float32} # (white occupied positions) - (black occupied positions), for mid and endgame
    score::Float32            # score of position in actual game (win/draw/loss)
    occupied::Vector{UInt32}  # only visit feautures that are non-zero
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
        for colour in [Scylla.WHITE, Scylla.BLACK]
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
    
    occupied = UInt32[]
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

"take a vector and split into a vector of vectors, each of length batch_size"
function batches(vec, batch_size)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    len = length(vec)
    return [@view vec[i:min(i + batch_size - 1, len)] for i in 1:batch_size:len]
end

"loop over all weights and nudge them in the direction that minimises the loss"
function gradient_descent!(
    weights::Vector{Float32},
    positions::Vector{Position};
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    k = K_VALUE,
)
    Random.shuffle!(positions)

    for batch in batches(positions, batch_size)
        gradient = zeros(Float32, length(weights))

        for position in batch
            y = position.score
            sigma_k = sigmoid(quality(position, weights), k)

            derivative = (2 * sigma_k - 1 - y) * (1 - sigma_k) * sigma_k

            for (linear_ind, jump_ind) in enumerate(position.occupied)
                gradient[jump_ind] += position.features[linear_ind] * derivative
            end
        end

        weights .-= (4 * k * learning_rate / length(batch)) * gradient
    end
end

function run_gradient_descent!(
    weights::Vector{Float32},
    positions::Vector{Position};
    epochs = 100,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    k = K_VALUE,
)
    for i in 1:epochs
        gradient_descent!(weights, positions; batch_size, learning_rate, k)
        println("Pass ", i, " completed, loss = ", mean_squared_error(positions, weights; k))
    end
end

function main()
    filename = "$(dirname(@__DIR__))/data/zurichess_serialize"
    #@time create_feature_data(filename)
    @time features = Serialization.deserialize(filename)
    weights = get_evaluation()

    run_gradient_descent!(weights, features)

    #@btime c = evaluate_all($features, $weights)
    #@btime mean_squared_error($features, $weights)
    #plot_error(features, weights)
    #plot_evaluation(features)
end

main()