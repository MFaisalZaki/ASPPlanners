%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- EXISTSTEP layer (prototype).
%
%   owns:     -- (a strategy layer: it reads no task fact of its own and is never
%                inferred from the task; it is loaded by naming it, `seq+existstep`)
%   requires: core (occurs/2, holds/3, precondition/3, postcondition/4)
%   excludes: numeric, temporal (see "Scope" below)
%
% Exists-step parallel semantics (Rintanen, Heljanko & Niemela, "Planning as
% satisfiability: parallel plans and algorithms for plan search", AIJ 2006).
%
% Under the sequential reading a horizon step holds one action, so the horizon
% the deepening search has to reach is the plan's *length*. Under the
% exists-step reading a step holds a SET of actions, legal when there exists
% some order in which they can be executed one after the other from the state
% before the step, and executing them in that order yields the state obtained
% by applying all their effects at once -- which is the state core's caused /
% inertia rules already compute. Every sequential plan is still a plan (one
% action per step), so nothing is lost; what is gained is that independent
% actions no longer cost a horizon step each, and the number of (ground-once,
% solve-many) horizons the search goes through drops with the plan's width.
%
% The order is FIXED up front rather than searched for: clingo's own total order
% on ground terms (`A < B` on the action terms) is the execution order of a step.
% That is Rintanen et al.'s linear-size encoding: for a fixed order, a set is
% legal iff
%   (D) no action DISABLES a later one -- writes a variable the later action
%       reads, to a value other than the one it reads -- and
%   (C) no two actions write one variable to different values.
% Why that suffices: executing the set in the fixed order, every variable an
% action reads is, when its turn comes, either untouched since the step's start
% or (by D) already carrying exactly the value it reads, so its preconditions
% still hold and the same conditional effects fire as core judged against the
% state before the step; and by (C) the final state is the simultaneous one.
%
% "Reads" is taken broadly, which is what keeps this sound on the shapes core
% and the disjunctive layer state: an action reads the variables of its
% preconditions, of the conditions of its conditional effects (they are judged
% at t-1 by core's caused/3, so an earlier action changing one would change
% which effects fire), and of every literal of its disjunctive conditions (an
% earlier action breaking the disjunct that held could leave none holding).
% "Writes" is every postcondition, conditional ones included whether or not
% their condition fires. Both over-approximate, which only ever forbids a
% parallel step that would have been fine; never the reverse.
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
% effects and the disjunctive layer. The numeric layer's effects and
% comparisons are not read here, so two actions incrementing one fluent would
% be allowed to share a step and the result would be a single caused/3 value
% rather than the sum; and the temporal layer relies on one snap action per
% happening. Both are therefore declared as exclusions in layers.py and
% refused at construction. Extending this to numeric tasks means adding a
% conservative rule -- an action writing a numeric fluent may not share a step
% with any other action that reads or writes it -- over numEffect/numAssign/
% num*Expr/numCondEffect* and the numTerm/3 vocabulary of the comparisons.
%
% Plan extraction. The planner serialises a step by the same term order, so
% the sequential plan it hands to the validator is the order argued above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pin the current program part; see the note in core.lp.
#program base.

% Switch core's one-action cap off; core states the choice as "at least one"
% plus a cap guarded by `not parallelSteps`.
parallelSteps.

#defined action/1.
#defined precondition/3.
#defined postcondition/4.
% Supplied by the disjunctive layer; the rules reading them are inert without it.
#defined orDisjunct/5.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% What each action reads and writes -- static, from the task facts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

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
% `forall`/`when` effects), and core settles that on its own with the add-wins
% rule -- it is not a conflict between actions and must stay legal.
written(V, Val, t) :- occurs(A, t), writes(A, V, Val).
twoValues(V, t) :- written(V, Val1, t), written(V, Val2, t), Val1 < Val2.
writesVar(A, V) :- writes(A, V, _).
writtenVar(V) :- writesVar(_, V).
twoWriters(V, t) :- writtenVar(V), 2 { occurs(A, t) : writesVar(A, V) }.
:- twoValues(V, t), twoWriters(V, t).
