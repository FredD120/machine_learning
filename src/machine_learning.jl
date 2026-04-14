using HDF5
using GLMakie
using Scylla
using .Threads
using BenchmarkTools

const SLICE = 1_000_000

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
    score::Int8               # score of position in actual game (win/draw/loss)
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

    score = 0
    if occursin("0-1", result)
        score = 1
    elseif occursin("1-0", result)
        score = -1
    end
    
    occupied = UInt32[]
    for (i, f) in enumerate(features)
        f != 0 ? push!(occupied, i) : nothing
    end

    return Position(features, score, occupied, Scylla.evaluate(board))
end

function get_features()
    data = readlines(pwd() * "\\data\\zurichess.epd")

    positions = Vector{Position}(undef, length(data))
    #=@threads=# for i in eachindex(data[1:SLICE])
        arr = split(data[i], "c9")
        positions[i] = Position(arr[1], arr[2])
    end
    return positions
end 

function dot(pos::Position, pst)
    count = Float32(0)
    for i in pos.occupied
        count += pos.features[i] * pst[i]
    end
    return count
end

function evaluate_all(pos, eval)
    count = 0.0
    for p in @view pos[1:SLICE]
        count += dot(p, eval)
    end
    return count
end

function main()
    @time pos = get_features()
    eval = get_evaluation()
    @btime c = evaluate_all($pos, $eval)
    #evaluate_all(pos, eval)
end
main()