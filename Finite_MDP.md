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




