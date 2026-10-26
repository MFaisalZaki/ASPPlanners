%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exists-step horizon encoding -- CORE layer (prototype).
%
%   owns:     type/1 constant/1 has/2 variable/1 boolean/1 action/1
%             precondition/3 postcondition/4 initialState/2 goal/2
%   requires: --
%
% The `exist` family is the `seq` family with one thing changed: a horizon step
% carries a SET of actions rather than one. Everything else -- the task facts,
% the multi-shot structure, effects, add-wins arbitration, inertia, the goal
% test, the disjunctive layer -- is the same, and the files are kept as copies
% here rather than shared, so that this directory is the whole encoding and a
% change to `seq` cannot silently change it (or the reverse).
%
% Exists-step parallel semantics (Rintanen, Heljanko & Niemela, "Planning as
% satisfiability: parallel plans and algorithms for plan search", AIJ 2006).
%
% Under the sequential reading the horizon the deepening search has to reach
% is the plan's *length*. Under the exists-step reading a step holds a set of
% actions, legal when there exists some order in which they can be executed one
% after the other from the state before the step, and executing them in that
% order yields the state obtained by applying all their effects at once --
% which is the state the caused / inertia rules below already compute. Every
% sequential plan is still a plan (one action per step), so nothing is lost;
% what is gained is that independent actions no longer cost a horizon step
% each, and the number of horizons the search goes through drops with the
% plan's width.
%
% The order is FIXED up front rather than searched for: clingo's own total
% order on ground terms (`A < B` on the action terms) is the execution order
% of a step. That is Rintanen et al.'s linear-size encoding: for a fixed order
% a set is legal iff
%   (D) no action DISABLES a later one -- writes a variable the later action
%       reads, to a value other than the one it reads -- and
%   (C) no two distinct actions write one variable to different values.
% Why that suffices: executing the set in the fixed order, every variable an
% action reads is, when its turn comes, either untouched since the step's start
% or (by D) already carrying exactly the value it reads, so its preconditions
% still hold and the same conditional effects fire as were judged against the
% state before the step; and by (C) the final state is the simultaneous one.
%
% "Reads" is taken broadly, which is what keeps this sound on the shapes core
% and the disjunctive layer state: an action reads the variables of its
% preconditions, of the conditions of its conditional effects (judged at t-1 by
% caused/3, so an earlier action changing one would change which effects
% fire), and of every literal of its disjunctive conditions (an earlier action
% breaking the disjunct that held could leave none holding). "Writes" is every
% postcondition, conditional ones included whether or not their condition
% fires. Both over-approximate, which only ever forbids a parallel step that
% would have been fine; never the reverse.
%
% Size. (C) is linear in the actions writing each variable. (D) is stated as a
% CHAIN rather than as action pairs: per (variable, value) somebody reads, the
% actions that read it or write past it are threaded in the fixed order
% (succ/4, computed once in base), and `broken/4` carries "an earlier action in
% this chain wrote past the value" down the chain one link per step. That is
% one rule per chain link per step, where the pairwise statement would be one
% constraint per pair per step. Computing succ/4 is quadratic per chain, but it
% is a one-off at grounding time over base facts (gringo evaluates the #min
% then and keeps only the links).
%
% Scope of the prototype. Boolean / multi-valued STRIPS with conditional
% effects, plus the disjunctive layer. The family has no numeric and no
% temporal layer: the numeric layer's effects and comparisons are not in the
% read/write analysis (two actions incrementing one fluent would be allowed to
% share a step and the result would be a single caused/3 value rather than the
% sum), and the temporal layer is built on one snap action per happening. A
% task that emits their facts is refused at construction (layers.py), not
% solved with a program that is silently wrong. Extending to numeric tasks
% means copying numeric.lp here and adding a conservative rule -- an action
% writing a numeric fluent may not share a step with any other action that
% reads or writes it -- over numEffect/numAssign/num*Expr/numCondEffect* and
% the numTerm/3 vocabulary of the comparisons.
%
% Plan extraction. The planner serialises a step by the same term order, so
% the sequential plan it hands to the validator is the order argued above.
%
% Multi-shot structure (iclingo-style), shared by every layer:
%   base     -- the task facts (added by the planner) and the initial state.
%   step(t)  -- at least one action at timestep t, their effects, and inertia.
%   check(t) -- the goal test at timestep t, gated on the external query(t)
%               so a deepening search can retire the test of horizon t and
%               move on to t+1 without regrounding earlier steps.
%
% The planner grounds base + check(0), then incrementally step(t) + check(t)
% for t = 1, 2, ... releasing query(t-1) each time. A fixed-horizon solve
% grounds base, step(1..h), check(h) in one go and sets query(h).
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
% Read by the interference analysis below: the literals of an action's (or an
% effect's) disjunctive conditions. Inert without the disjunctive layer.
#defined orDisjunct/5.

#show occurs/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% What each action reads and writes -- static, from the task facts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

writes(A, V, Val) :- postcondition(A, _, V, Val).

reads(A, V, Val) :- action(A), precondition(A, V, Val).
reads(A, V, Val) :- postcondition(A, Eff, _, _), precondition(Eff, V, Val).
% Every literal of every disjunct of an action's disjunctive conditions, nested
% groups included -- the disjunctive layer keys them all by the action term.
reads(A, V, Val) :- action(A), orDisjunct(A, _, _, V, Val).
reads(A, V, Val) :- postcondition(A, Eff, _, _), orDisjunct(Eff, _, _, V, Val).

% A writes V past the value Val that somebody reads.
readValue(V, Val) :- reads(_, V, Val).
writesPast(A, V, Val) :- writes(A, V, Other), readValue(V, Val), Other != Val.

% The chain for (V, Val): the actions that read it or write past it, threaded
% in the fixed order. `B = #min {...}` is evaluated by the grounder (every atom
% involved is a base fact); the trailing `touches` guard drops the last link,
% whose minimum over the empty set is #sup.
touches(V, Val, A) :- reads(A, V, Val).
touches(V, Val, A) :- writesPast(A, V, Val).
succ(V, Val, A, B) :- touches(V, Val, A), B = #min { C : touches(V, Val, C), C > A }, touches(V, Val, B).

% For (C): the variables each action writes at all.
writesVar(A, V) :- writes(A, V, _).
writtenVar(V) :- writesVar(_, V).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Establish initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

holds(Variable, Value, 0) :- initialState(Variable, Value).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

% A non-empty SET of actions per step; which sets are legal is stated below.
1 {occurs(Action, t) : action(Action)}.

% (D) broken(V, Val, B, t): some action strictly before B in the (V, Val) chain
% occurs at t and writes V past Val. Carried down the chain link by link, so B
% never sees its own write -- an action that reads V = x and sets V := y, as
% every `move` does, is not in conflict with itself.
broken(V, Val, B, t) :- succ(V, Val, A, B), occurs(A, t), writesPast(A, V, Val).
broken(V, Val, B, t) :- succ(V, Val, A, B), broken(V, Val, A, t).
:- occurs(B, t), reads(B, V, Val), broken(V, Val, B, t).

% (C) Two DISTINCT actions may both write a variable only if everything written
% to it at the step is one value. Stated over the values written and a count of
% the writers rather than over action pairs: `written/3` is linear in the
% writers, `twoWriters/2` is one aggregate per variable per step, and a variable
% has few values (two, for a boolean). The "distinct" matters: one action may
% set and clear the same fluent through two conditional effects whose
% conditions both hold (lift's `stop` boards and un-boards through two
% `forall`/`when` effects), and the add-wins rule below settles that on its
% own -- it is not a conflict between actions and must stay legal.
written(V, Val, t) :- occurs(A, t), writes(A, V, Val).
twoValues(V, t) :- written(V, Val1, t), written(V, Val2, t), Val1 < Val2.
twoWriters(V, t) :- writtenVar(V), 2 { occurs(A, t) : writesVar(A, V) }.
:- twoValues(V, t), twoWriters(V, t).

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
