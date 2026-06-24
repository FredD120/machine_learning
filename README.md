# Evaluation Function

There are 6 piece types, 64 squares and a mid-game and end-game phase.
This means that there are 768 total features describing a chess
position. A linear function can be used to evaluate the quality of a
chess position $q$ (whether white is winning or losing and by how much
in centi-pawns), by defining a vector of weights that assign points to
certain pieces being in certain positions:

$$q = \sum_{i=1}^{768}w_if_i$$

This is clearly a dot product between the weight vector, which is the
same for all chess positions, and the feature vector, which contains all
of the information in a chess position. A simple way to construct the
feature vector in the absence of separate mid/end-game phases would be
to assign $+1$ to a white piece being on a certain square and $-1$ to a
black piece being on the corresponding square mirrored across the
central horizontal by the transformation $rank \rightarrow 9-rank$. This
ensures that a position's evaluation $q$ becomes $-q$ when the entire
board is reflected across the central horizontal line and the colours of
all pieces are inverted.\
To take the phase of the game into account, we linearly interpolate
between the score that each piece at a particular square would get at
the start of the game with the equivalent score in the end-game. The
phase can be determined in a number of ways, but a simple way to do it
is to count the total number of pieces on the board. The phase of the
game $p$ starts at $1$ in the early game and tends linearly to $0$ in
the end-game, completely determined by number of pieces $n$:
$$p = \text{max}\left(0,\text{min}\left(\frac{n-8}{16},1\right)\right)$$

To implement this idea in the feature vector, now each mid-game feature
is assigned a value of $p$ and $-p$, while each end-game feature is
assigned a value of $1-p$ and $p-1$, for white and black pieces
respectively. This ensures that when evaluating a late-game position,
the late-game weights are heavily favoured, while a completely different
set of weights can be used to evaluate early-game positions. Now all
that remains is to determine the values of those weights.

# Gradient Descent

If we could search any chess position with unlimited computational
resources until we found an end state - win, loss or draw - we would
know with certainty the quality of that position: $+\infty$, $-\infty$
and $0$ respectively. Since we can't do this, Eq.
(1) needs to be a good-enough approximation of this
'true' score. The difference between the score of a position predicted
by this evaluation function and the actual score reached by playing the
game out near perfectly from that position (obtained from games played
by high level players/engines), can be used to determine the accuracy of
the evaluation function for this position. By summing this difference
over many positions, we can determine how good the evaluation function
is - this is called the loss function. For numerical stability, it is
better for the score of each position to be between $-1$ and $1$ rather
than being determined by a value in centi-pawns. Therefore the true
score of the $j$-th position $y_j$ is $1$/$-1$/$0$ for a win/loss/draw
and the estimated score $\hat{y}_j$ is normalised by a sigmoid function:
$$\hat{y}_j = 2\,\sigma_k(q_j) - 1 \quad \text{where} \quad \sigma_k(q_j) = \frac{1}{1+e^{-kq_j}}$$

Here $k$ determines the width of the sigmoid function, which can be
tuned to improve the optimisation procedure. Since the quality $q_j$ is
typically given in centi-pawns, $1/k$ should be approximately the
centi-pawn advantage that guarantees winning a game, ie. about 500.\
For a dataset of $N$ positions, the loss function $L$ is defined as the
mean squared error between prediction and reality:
$$L = \frac{1}{N}\sum_{j=1}^N(y_j-\hat{y}_j)^2$$

This loss function now becomes a target, if we can find a way to
minimise this value across the whole training dataset, we should have a
locally optimal evaluation function. The optimisation function we will
use is gradient descent. This is a recursive process where on every pass
we adjust each of the weights that determines the loss function in the
direction that minimises the loss function.

$$w_i \leftarrow w_i - \alpha\frac{\partial L}{\partial w_i}$$

Here, $\alpha$ is the learning rate, which determines how quickly the
optimisation process approaches a local minimum, but must be tuned so
that it doesn't overshoot the minimum. To calculate the derivative of
$L$ with respect to the $i$-th weight $w_i$, we first calculate the
derivative of the mean squared error:
$$\frac{\partial L}{\partial w_i} = \frac{2}{N}\sum_{j=1}^N (\hat{y}_j - y_j)\frac{\partial \hat{y}_j}{\partial w_i}$$

Next, take the derivative of the normalised quality evaluation:
$$\frac{\partial \hat{y}_j}{\partial w_i} = 2\frac{\partial \sigma_k(q_j)}{\partial q_j} \cdot \frac{\partial q_j}{\partial w_i} = 2k\sigma_k(q_j)\cdot(1-\sigma_k(q_j))\cdot \frac{\partial q_j}{\partial w_i}$$

Finally, the derivative of the quality with respect to a weight is
trivial:
$$\frac{\partial q_j}{\partial w_i} = \frac{\partial}{\partial w_i} \sum_{m=1}^{768}w_mf_{m,j} = f_{i,j}$$

Where $f_{i,j}$ is the $i$-th feature of the $j$-th position. This
derivative picks out only the relevant feature to be tuned. If this
feature is equal to zero for a given position, then the corresponding
weight is not updated, since it has no effect on the score of the
position. Taking all of this together:
$$\frac{\partial L}{\partial w_i} = \frac{4k}{N}\sum_{j=1}^N (2\,\sigma_k(q_j) - 1 - y_j)\cdot\sigma_k(q_j)\cdot(1-\sigma_k(q_j))\cdot f_{i,j}$$
