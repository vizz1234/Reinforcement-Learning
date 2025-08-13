<h1>Finite Markov Decision Process</h1>

<p>Now that we have good understanding of the basics of RL and MAB algorithms, let's move on to solving full RL problem (remember that MAB are not full RL problem).</p>

<h2>Agent Environment Interface</h2>

<p>
Reinforcement Learning (RL) is about learning from interaction to achieve a goal. 
The <strong>agent</strong> is the learner and decision-maker, while the <strong>environment</strong> contains everything outside the agent. 
They interact continuously: the agent selects actions, the environment responds with new states and rewards.
</p>

<h3>Interaction Process</h3>
<ol>
  <li>At time step <em>t</em>, the agent observes the <strong>state</strong> <em>S<sub>t</sub></em>.</li>
  <li>It chooses an <strong>action</strong> <em>A<sub>t</sub></em> from the available set <em>A(S<sub>t</sub>)</em>.</li>
  <li>The environment returns:
    <ul>
      <li>A new state <em>S<sub>t+1</sub></em></li>
      <li>A numerical <strong>reward</strong> <em>R<sub>t+1</sub></em></li>
    </ul>
  </li>
</ol>

<p>
The agent’s behavior is determined by its <strong>policy</strong> π<sub>t</sub>, mapping states to probabilities of actions. 
The goal is to maximize long-term rewards.
</p>

<h3>Key Features</h3>
<ul>
  <li><strong>States</strong>: Can be sensory inputs, abstract descriptions, or memory-based information.</li>
  <li><strong>Actions</strong>: May be low-level controls, high-level decisions, or even mental/computational choices.</li>
  <li><strong>Rewards</strong>: External signals defining the goal, beyond the agent’s direct control.</li>
  <li><strong>Agent–Environment Boundary</strong>: Anything the agent cannot change arbitrarily is part of the environment.</li>
</ul>

<h3>Flexibility of the Framework</h3>
<ul>
  <li>Time steps need not match real time — they can be decision stages.</li>
  <li>States and actions vary by task and representation choices strongly affect performance.</li>
  <li>Multiple agents may exist at different levels within a system (e.g., high-level decision-maker vs. low-level controller).</li>
  <li>The framework applies to physical, mental, or abstract decision-making processes.</li>
</ul>

<h3>Core RL Signals</h3>
<p>
Any RL problem can be reduced to three signals exchanged between agent and environment:
</p>
<ol>
  <li><strong>Actions</strong>: The agent’s choices.</li>
  <li><strong>States</strong>: The basis for making those choices.</li>
  <li><strong>Rewards</strong>: The goals to maximize.</li>
</ol>
<h2>3.2 Goals and Rewards</h2>
<p>
In reinforcement learning, an agent’s purpose is expressed through a <strong>reward signal</strong> provided by the environment. At each time step, the agent receives a scalar reward <code>R<sub>t</sub></code>. The aim is not just to maximize immediate reward, but to maximize the <strong>expected cumulative reward</strong> over time.
</p>
<blockquote>
Reward Hypothesis: All goals and purposes can be framed as maximizing the expected value of the cumulative sum of a scalar reward signal.
</blockquote>
<p>
The reward signal is a way of telling the agent <em>what</em> to achieve, not <em>how</em> to achieve it. If the reward function is poorly designed, the agent may learn behaviors that achieve high reward without meeting the designer’s true goals.
</p>
<p>Examples of reward design:</p>
<ul>
  <li>Robot walking: reward proportional to forward distance covered.</li>
  <li>Maze escape: −1 reward per time step until escape, encouraging speed.</li>
  <li>Soda can collection: +1 per can collected, 0 otherwise, negative for collisions.</li>
  <li>Chess or checkers: +1 for win, −1 for loss, 0 for draw or nonterminal positions.</li>
</ul>
<p>
Rewarding subgoals can backfire. For example, a chess agent rewarded for capturing pieces might sacrifice the game to capture more pieces, missing the real objective of winning.
</p>
<p>
In reinforcement learning, rewards are defined in the <strong>environment</strong>, not within the agent. Even internal states—such as a robot’s energy levels or limb positions—are considered part of the environment for learning purposes. The agent’s boundary is drawn at the limit of its control, ensuring it cannot simply assign itself rewards directly.
</p>
<p>
This separation ensures the reward reflects outcomes the agent cannot trivially manipulate. While the external reward defines the ultimate goal, the agent may still construct internal reward-like signals to aid in learning.
</p>
<h2>3.3 Returns</h2>
<p>
An agent’s goal is to maximize its <strong>expected return</strong>—a function of the future rewards it will receive. If rewards after time <em>t</em> are <code>R<sub>t+1</sub>, R<sub>t+2</sub>, ...</code>, then the return <code>G<sub>t</sub></code> is defined as a specific function of this sequence.
</p>

<h3>Episodic Tasks</h3>
<p>
In tasks with a natural end point (episodes), the return is simply the sum of rewards until the terminal state:
</p>
<pre><code>G<sub>t</sub> = R<sub>t+1</sub> + R<sub>t+2</sub> + ... + R<sub>T</sub></code></pre>
<ul>
  <li><strong>Definition:</strong> Interaction breaks into finite sequences ending in a terminal state, then restarts from a standard or sampled starting state.</li>
  <li><strong>Examples:</strong> A play of chess, a trip through a maze, one complete game in sports.</li>
  <li><strong>Notation:</strong> <code>S</code> = nonterminal states, <code>S⁺</code> = all states including terminal.</li>
</ul>

<h3>Continuing Tasks</h3>
<p>
In tasks that go on indefinitely (e.g., process control, autonomous robots), summing rewards directly can be infinite. Instead, use <strong>discounted return</strong>:
</p>
<pre><code>G<sub>t</sub> = R<sub>t+1</sub> + γR<sub>t+2</sub> + γ²R<sub>t+3</sub> + ... 
           = Σ<sub>k=0</sub><sup>∞</sup> γ<sup>k</sup> R<sub>t+k+1</sub></code></pre>
<ul>
  <li><strong>γ (discount rate)</strong> is between 0 and 1.</li>
  <li>γ = 0 → agent is <em>myopic</em> (only immediate rewards matter).</li>
  <li>γ near 1 → agent is <em>farsighted</em> (future rewards valued strongly).</li>
  <li>Ensures finite returns when rewards are bounded.</li>
</ul>
<p>
Maximizing immediate rewards may reduce long-term gains, so the agent must balance short- and long-term outcomes.
</p>

<h3>Example: Pole-Balancing</h3>
<p>
A cart must keep a hinged pole upright by applying forces along a track. Failure occurs if:
</p>
<ul>
  <li>The pole tilts past a set angle from vertical, or</li>
  <li>The cart moves off the track.</li>
</ul>
<p>
After failure, the pole is reset to vertical. Two formulations:
</p>
<ul>
  <li><strong>Episodic:</strong> Reward +1 per time step without failure; return = steps until failure.</li>
  <li><strong>Continuing:</strong> Reward 0 per step, −1 on failure; return ≈ −γ<sup>K</sup> where <code>K</code> = steps before failure.</li>
</ul>
<p>
In both cases, maximizing the return means keeping the pole balanced for as long as possible.
</p>
<h2>3.4 Unified Notation for Episodic and Continuing Tasks</h2>
<p>
Reinforcement learning tasks may be:
</p>
<ul>
  <li><strong>Episodic:</strong> Interaction breaks into finite episodes with terminal states.</li>
  <li><strong>Continuing:</strong> Interaction proceeds indefinitely without a terminal state.</li>
</ul>
<p>
Episodic tasks are mathematically simpler because each action affects only a finite number of future rewards. However, to discuss both cases consistently, we adopt a unified notation.
</p>

<h3>Notation for Episodic Tasks</h3>
<p>
In episodic tasks, each episode starts at <code>t = 0</code>. Full notation includes the episode index <code>i</code>:
</p>
<ul>
  <li><code>S<sub>t,i</sub></code> = state at time <em>t</em> of episode <em>i</em></li>
  <li><code>A<sub>t,i</sub></code> = action at time <em>t</em> of episode <em>i</em></li>
  <li><code>R<sub>t,i</sub></code> = reward at time <em>t</em> of episode <em>i</em></li>
</ul>
<p>
In practice, we drop the episode index when it is not important, writing simply <code>S<sub>t</sub></code>, <code>A<sub>t</sub></code>, <code>R<sub>t</sub></code>.
</p>

<h3>Unifying Episodic and Continuing Returns</h3>
<p>
Return definitions:
</p>
<ul>
  <li><strong>Episodic:</strong> finite sum of rewards (Eq. 3.1)</li>
  <li><strong>Continuing:</strong> infinite discounted sum (Eq. 3.2)</li>
</ul>
<p>
We unify them by treating episode termination as entering a <strong>special absorbing state</strong> that:
</p>
<ul>
  <li>Transitions only to itself</li>
  <li>Produces rewards of 0 forever after</li>
</ul>

<h4>Example:</h4>
<pre>
S0 --(+1)--> S1 --(+1)--> S2 --(+1)--> [Terminal State]
                                   ↓
                               (0) → [Terminal State]
                                   ↓
                               (0) → [Terminal State]
                                   ...
</pre>
<p>
Starting at <code>S<sub>0</sub></code>, the reward sequence is: +1, +1, +1, 0, 0, 0, …  
Summing over the first <code>T</code> rewards (<code>T = 3</code>) or the infinite sequence gives the same return. This also holds when discounting is applied.
</p>

<h3>General Return Formula</h3>
<p>
With this convention, the return can be expressed for both episodic and continuing tasks as:
</p>
<pre><code>G<sub>t</sub> = Σ<sub>k=0</sub><sup>T−t−1</sup> γ<sup>k</sup> R<sub>t+k+1</sub></code></pre>
<ul>
  <li><code>T</code> can be finite (episodic) or infinite (continuing).</li>
  <li><code>γ</code> can be 1 if episodes always terminate.</li>
  <li>We avoid both <code>T = ∞</code> and <code>γ = 1</code> unless the sum is finite by other means.</li>
</ul>

<p>
This unified notation lets us write formulas once and apply them to both task types, emphasizing their close parallels.
</p>
<h2>3.5 The Markov Property</h2>

<p>
In reinforcement learning, the agent chooses actions based on a signal from the environment called the <strong>state</strong>. Here, “state” means whatever information is available to the agent, provided by some preprocessing system (considered part of the environment). We focus not on designing this signal, but on deciding actions given it.
</p>

<p>
A state signal can include immediate sensations (e.g., sensor readings) but may also contain processed information built from past sensations. For example:
</p>
<ul>
  <li>Looking around a scene to build a detailed mental picture</li>
  <li>Remembering an object after looking away</li>
  <li>Interpreting the meaning of the word “yes” based on a previous question</li>
  <li>Calculating velocity from two position measurements</li>
</ul>

<p>
The state need not tell the agent <em>everything</em>—hidden information exists in most environments. For example, in blackjack, the agent cannot know the next card; a phone-answering agent cannot know the caller in advance; a paramedic cannot instantly know internal injuries. The agent should not be faulted for lacking information it never sensed—only for forgetting something it already knew.
</p>

<h3>The Ideal: Markov States</h3>
<p>
Ideally, the state summarizes past sensations so that all relevant information for decision-making is retained—more than just immediate sensations, but never more than the complete history. If a state retains all relevant information, it has the <strong>Markov property</strong>.
</p>
<p>
Examples of Markov states:
</p>
<ul>
  <li>A checkers board position — captures everything relevant about the game’s future.</li>
  <li>The position and velocity of a cannonball — enough to predict its flight.</li>
</ul>

<p>
This is sometimes called the <em>independence of path</em> property: the meaning of the state is independent of the exact sequence of events that led to it.
</p>

<h3>Formal Definition</h3>
<p>
In the most general (causal) case, the next state and reward may depend on the full history:
</p>
<pre>
Pr{R<sub>t+1</sub> = r, S<sub>t+1</sub> = s' | S<sub>0</sub>, A<sub>0</sub>, R<sub>1</sub>, ..., S<sub>t</sub>, A<sub>t</sub>}   (3.4)
</pre>
<p>
If the state has the Markov property, the next state and reward depend only on the current state and action:
</p>
<pre>
p(s', r | s, a) = Pr{R<sub>t+1</sub> = r, S<sub>t+1</sub> = s' | S<sub>t</sub> = s, A<sub>t</sub> = a}   (3.5)
</pre>
<p>
A state is Markov if (3.5) = (3.4) for all possible values. In a Markov environment, iterating (3.5) lets us predict the future as well as if we had the full history, and Markov states provide the best possible basis for choosing actions.
</p>

<h3>Practical Considerations</h3>
<p>
In practice, states are often <em>approximations</em> to Markov states. We want them to be good predictors of future rewards, and (if learning a model) future states. Even if not strictly Markov, such states can still work well, and theory developed for the Markov case often applies approximately.
</p>

<h3>Example 3.5: Pole-Balancing State</h3>
<p>
In the cart–pole task, a state would be Markov if it specified exactly:
</p>
<ul>
  <li>Position and velocity of the cart</li>
  <li>Angle of the pole</li>
  <li>Angular velocity of the pole</li>
</ul>
<p>
This would be enough to exactly predict future motion in an idealized system. In reality, sensors introduce noise and delay, and unmodeled effects (pole bending, bearing temperature, backlash) break the Markov property.
</p>
<p>
Nevertheless, positions and velocities often work well in practice. Early studies even used a coarse state signal—dividing positions into regions (“left,” “middle,” “right”) and similarly quantizing other variables. Despite being non-Markov, this representation was sufficient to solve the task and may have helped learning by ignoring irrelevant detail.
</p>

<h3>Example 3.6: Draw Poker</h3>
<p>
In draw poker, each player knows only their own hand. The state for a player should not include the other players’ cards or the deck’s contents—these cannot be determined from past observations in a fair game.
</p>
<p>
Useful state information includes:
</p>
<ul>
  <li>Your own cards</li>
  <li>Bets made by other players</li>
  <li>Number of cards each opponent drew</li>
  <li>Behavioral tendencies of opponents (e.g., bluffing style, playing conservatively, changes in play late at night)</li>
</ul>
<p>
While past interactions with players matter, it is impractical to remember everything; good players focus on key clues. Thus, human poker states are non-Markov, and decisions imperfect—yet still effective.
</p>

<p>
In summary, the Markov property is central to RL theory: decisions and values are assumed to depend only on the current state. Even when states are non-Markov, aiming for good approximations to Markov states improves performance.
</p>







