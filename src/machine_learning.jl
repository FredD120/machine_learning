using HDF5
using GLMakie
using Scylla
using Threads

struct Position
    features::Vector{Float32}
    score::Int8

    function Position(f::Vector{Float32}, s::Int8)
        return New(f, s)
    end
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

    return Position(features, score)
end

function get_features()
    data = readlines(pwd() * "\\data\\zurichess.epd")

    positions = Vector{Position}(undef, length(data))
    Threads.@threads for i in eachindex(data[1:100])
        arr = split(data[i], "c9")
        positions[i] = Position(arr[1], arr[2])
    end
    return positions
end 
pos = get_features()
println(pos[2])