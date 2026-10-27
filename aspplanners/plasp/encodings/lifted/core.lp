%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lifted horizon encoding -- CORE layer.
%
%   owns:     type/1 constant/1 has/2 variable/1 boolean/1 schema/1 param/3
%             initialState/2 goal/2
%   requires: -- (the per-schema step rules the encoder emits; see below)
%
% The seq core with the action choice lifted (after Holler & Behnke, ICAPS
% 2022; see aspplanners/plasp/lifted.py for the whole story). A step chooses an
% action SCHEMA and one object per parameter position instead of a ground
% action, so no ground action is ever named: there is no action/1, no
% occurs/2, and no precondition/3 or postcondition/4 fact. In their place the
% encoder emits, into this same step(t) part, one rule per precondition literal
% and per effect of each schema, joining only the arg/4 atoms of the parameters
% that literal mentions:
%
%   :- schema("drive", t), arg("drive", 1, T, t), arg("drive", 2, A, t),
%      not holds(variable(("at",T,A)), value(variable(("at",T,A)), true), t-1).
%   caused(variable(("at",T,B)), value(variable(("at",T,B)), true), t) :-
%      schema("drive", t), arg("drive", 1, T, t), arg("drive", 3, B, t).
%
% Those rules feed caused/3, and from there on -- add-wins arbitration,
% inertia, the goal test -- this file is seq's core unchanged. One action per
% step, as in seq. Classical tasks only in this first version: the numeric,
% disjunctive and temporal layers would each need restating per schema, and a
% task emitting their facts is refused at construction with that reason.
%
% Multi-shot structure (iclingo-style):
%   base     -- the task facts (added by the planner) and the initial state.
%   step(t)  -- the schema and argument choice at timestep t, the encoder's
%               per-schema rules, their effects, and inertia.
%   check(t) -- the goal test at timestep t, gated on the external query(t)
%               so a deepening search can retire the test of horizon t and
%               move on to t+1 without regrounding earlier steps.
%
% The planner grounds base + check(0), then incrementally step(t) + check(t)
% for t = 1, 2, ... releasing query(t-1) each time. A fixed-horizon solve
% grounds base, step(1..h), check(h) in one go and sets query(h).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Every layer file opens by pinning the current program part, so that loading the
% layers as separate files and concatenating them into one program (what
% PLASPPlanner.lp_program does) are the same program. Without it the statements
% above a file's first `#program` would inherit the part the previous file
% happened to end in.
#program base.

% Predicates that may legitimately be absent from a task's facts.
#defined schema/1.
#defined param/3.
#defined goal/2.
% caused/3 is derived only by the encoder's per-schema step rules, which are
% added to step(t) by the planner rather than read from this file; the step
% atoms themselves do not exist until the first step(t) part is grounded
% (horizon 0), and a task without objects declares no has/2 at all.
#defined caused/3.
#defined schema/2.
#defined arg/4.
#defined has/2.

#show schema/2.
#show arg/4.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Establish initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

holds(Variable, Value, 0) :- initialState(Variable, Value).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

% One schema per step, and one object of the declared type per parameter
% position of the chosen schema. The encoder's per-schema rules then read
% `schema(S, t)` and `arg(S, I, O, t)`: a precondition is a constraint on the
% state before the step under the arguments its literal mentions, an effect a
% caused/3 rule under the same, with the parameters it does not mention left
% out of the join altogether.
1 {schema(S, t) : schema(S)} 1.
1 {arg(S, I, O, t) : has(O, Type)} 1 :- schema(S, t), param(S, I, Type).

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
