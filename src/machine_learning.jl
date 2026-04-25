using HDF5
using GLMakie
using Scylla
using BenchmarkTools
import .Threads
import Serialization

const SLICE = 1_428_000 # max size zurichess = 1_428_000
const K_VALUE = 3.0 / 500 # hyperparameter in sigmoid function

"Retrieve piece square tables from file"
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
            for pos in board.pieces[Scylla.colour_piece_id(colour, type)]
                sign = Scylla.sgn(colour)
                piece_ind = Scylla.side_index(colour, pos) + 1
                board_ind = (type - 1) * 64

                features[board_ind + piece_ind] += sign * mg_phase
                features[board_ind + piece_ind + 384] += sign * eg_phase
            end
        end
    end

    score = 0.5
    if occursin("1-0", result) # white wins
        score = 1.0
    elseif occursin("0-1", result) # black wins
        score = 0.0
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

function get_features()
    data = readlines(pwd() * "\\data\\zurichess.epd")

    positions = Vector{Position}(undef, length(data))
    for i in eachindex(data[1:SLICE])
        arr = split(data[i], "c9")
        positions[i] = Position(arr[1], arr[2])
    end
    return positions
end 

function dot(pos::Position, pst)
    count = Float32(0)
    for (linear_index, jump_index) in enumerate(pos.occupied)
        count += pos.features[linear_index] * pst[jump_index]
    end
    #@assert floor(Int16, count) == pos.original_eval
    return count
end

sigmoid(x, k) = 1 / (1 + exp(-k * x))

calculate_loss(feature, eval, k) = sigmoid(dot(feature, eval), k) - feature.score

function mean_squared_error(feature_set, eval)
    error = Float32(0.0)
    for feature in @view feature_set[1:SLICE]
        loss = calculate_loss(feature, eval, K_VALUE)
        error += loss * loss
    end
    return error / SLICE
end

function create_feature_data(filename)
    features = get_features()
    Serialization.serialize(filename, features)
end

function plot_loss(feature_set, eval, BUCKET_COUNT = 100)
    buckets = zeros(BUCKET_COUNT)
    for feature in feature_set
        loss = calculate_loss(feature, eval, K_VALUE)
        ind = floor(Int16, (BUCKET_COUNT - 1) * (loss + 1) / 2) + 1
        buckets[ind] += 1
    end
    xs = [-1.0 + 2 * i / BUCKET_COUNT for i in 1:BUCKET_COUNT]
    barplot(xs, buckets)
end

function plot_evaluation(feature_set, BUCKET_COUNT = 100)
    buckets = zeros(BUCKET_COUNT)
    for feature in feature_set
        ind = floor(Int16, (BUCKET_COUNT - 1) * (1 + feature.original_eval / 900) / 2)
        ind = min(BUCKET_COUNT, max(1, ind + 1))
        buckets[ind] += 1
    end
    barplot(buckets)
end

function main()
    filename = "$(dirname(@__DIR__))/data/zurichess_serialize"
    #@time create_feature_data(filename)
    @time feature_set = Serialization.deserialize(filename)
    eval = get_evaluation()
    #@btime c = evaluate_all($feature_set, $eval)
    #@btime mean_squared_error($feature_set, $eval)
    plot_loss(feature_set, eval)
    #plot_evaluation(feature_set)
end
main()