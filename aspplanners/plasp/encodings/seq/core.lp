%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- CORE layer.
%
%   owns:     type/1 constant/1 has/2 variable/1 boolean/1 action/1
%             precondition/3 postcondition/4 initialState/2 goal/2
%   requires: --
%
% Core is loaded by every profile and is always loaded first, so the program
% parts and the `query(t)` external below are declared exactly once.
%
% Multi-shot structure (iclingo-style), shared by every layer:
%   base     -- the task facts (added by the planner) and the initial state.
%   step(t)  -- exactly one action at timestep t, its effects, and inertia.
%   check(t) -- the goal test at timestep t, gated on the external query(t)
%               so a deepening search can retire the test of horizon t and
%               move on to t+1 without regrounding earlier steps.
%
% The planner grounds base + check(0), then incrementally step(t) + check(t)
% for t = 1, 2, ... releasing query(t-1) each time. A fixed-horizon solve
% grounds base, step(1..h), check(h) in one go and sets query(h).
%
% Timesteps double as SMTPlan-style *happenings* when the task has durative
% actions; see the temporal layer.
%
% Every layer contributes rules to these same three parts: clingo merges
% `#program` parts of the same name across the files handed to it, so the
% composed program is exactly the concatenation of the loaded layers. Layer
% files reference each other's predicates only through `#defined`, which is
% what makes an unloaded layer inert rather than an error.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Every layer file opens by pinning the current program part, so that loading the
% layers as separate files and concatenating them into one program (what
% PLASPPlanner.lp_program does) are the same program. Without it the statements
% above a file's first `#program` would inherit the part the previous file
% happened to end in.
#program base.

% Predicates that may legitimately be absent from a task's facts.
% occurs/2 is absent until the first step(t) part is grounded (horizon 0).
#defined occurs/2.
#defined action/1.
#defined goal/2.
#defined precondition/3.
#defined postcondition/4.

% The conditional-effect rule below reads the disjunctive layer's vocabulary so
% that an effect condition may carry a disjunction. This is the one place core
% points *up* at a layer, and it stays inert without it: a conditional literal
% whose condition (`orGroup(Effect, G)`) has no instances is vacuously true.
#defined orGroup/2.
#defined orGroupHolds/3.

#show occurs/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Establish initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

holds(Variable, Value, 0) :- initialState(Variable, Value).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

1 {occurs(Action, t) : action(Action)} 1.

% Check preconditions
:- occurs(Action, t), precondition(Action, Variable, Value), not holds(Variable, Value, t - 1).

% Apply effects. A conditional effect fires when every precondition/3 atom of its
% Effect term holds at t-1 and every disjunctive condition it carries does too --
% the orGroup/2 vocabulary of the disjunctive layer, keyed by the effect term
% instead of an action (the constraint on orGroup/2 there is guarded by occurs/2,
% so an effect's groups are only ever read here).
caused(Variable, Value, t) :- occurs(Action, t), postcondition(Action, Effect, Variable, Value), holds(VariablePre, ValuePre, t - 1) : precondition(Effect, VariablePre, ValuePre); orGroupHolds(Effect, G, t) : orGroup(Effect, G).

% Inertia
modified(Variable, t) :- caused(Variable, Value, t).

% An action may set and clear the same fluent at one step -- two conditional
% effects whose conditions both hold, say, which is the shape the ADL domains
% reach through `forall`/`when` over the same predicate. PDDL settles it the
% same way STRIPS does: the add wins. Arbitrating it here rather than dropping
% one of the effects up front is what keeps a *conditional* clear alive for the
% states its condition picks out; without it both values would hold at once,
% which is not a state at all. A non-boolean value has no such rule -- two
% effects assigning one numeric fluent different values is a malformed task,
% not a race with a winner.
holds(Variable, Value, t) :- caused(Variable, Value, t), Value != value(Variable, false).
holds(Variable, value(Variable, false), t) :- caused(Variable, value(Variable, false), t),
                                              not caused(Variable, value(Variable, true), t).
holds(variable(V), Value, t) :- holds(variable(V), Value, t - 1), not modified(variable(V), t).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verify that goal is met
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

#external query(t).

:- query(t), goal(Variable, Value), not holds(Variable, Value, t).
