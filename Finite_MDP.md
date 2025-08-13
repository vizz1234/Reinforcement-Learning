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






