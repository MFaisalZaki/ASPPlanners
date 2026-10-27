%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- PDR layer: the one-step query program of
% property-directed reachability.
%
%   owns:     -- (not a feature layer: loaded by the planner for search='pdr',
%                never selected from the task)
%   requires: core (holds/3, occurs/2, query/1); the disjunctive layer's rules
%             apply unchanged when the task needs it
%
% Property-directed reachability (Bradley 2011; for planning, Suda, "Property
% Directed Reachability for Automated Planning", JAIR 2014) does not unroll
% the horizon. It keeps LAYERS of blocked cubes, B_1, B_2, ..., where a cube
% in B_i is a partial state none of whose completions can reach the goal
% within i steps, and works on OBLIGATIONS "state s must reach the goal in i
% steps": find an action from s whose successor is not blocked at i-1, or, if
% none exists, learn a REASON -- a subset of s's literals that already rules
% every action out -- and block it at i. Layers are never thrown away, which
% is the opposite of what horizon deepening does with everything it learned
% through the goal; and when a layer stops growing the task is proven
% unsolvable, which deepening can never do.
%
% Every PDR query is a question about ONE step, so one ground program answers
% all of them: this file turns the state at time point 0 into a free choice
% that each query fixes through solver assumptions, lets step(1) be the seq
% encoding's own step, and gates the goal test at time point 1 (core's
% check(1)) and the blocked cubes at time point 1 behind atoms the query
% switches. The reason extraction, the hard part of any PDR, is clingo's
% unsatisfiable core over the state assumptions, for free and already
% generalised; the planner's PDR loop (aspplanners/plasp/pdr.py) does the
% rest.
%
% How the pieces fit:
%
%   * The planner adds the task facts with initialState/2 renamed to
%     pdrInitialState/2 for every fluent some action writes, so core's
%     `holds(V, Val, 0) :- initialState(V, Val)` fixes nothing but the STATIC
%     fluents (whose facts keep their name: the encoder folds static
%     preconditions into the action signatures as initialState/2 body atoms,
%     which have to stay true). The state at 0 is the choice below, with the
%     statics pinned to their initial value by assumption in every query. The
%     initial state is then just the first obligation's state.
%   * Boolean variables are free at time 0. A query assumes, for every one of
%     them, whether its `true` atom holds (so the state is total, and so that
%     a reason may name a variable's FALSITY); a core is a subset of those
%     assumptions, and dropping a variable from it leaves that variable free
%     -- the very thing that makes the reason a cube rather than one state.
%     The false side is derived, which keeps negative conditions working.
%   * A step may also do nothing: pdrNoop below is an action with no
%     condition and no effect, so that a query at level i also admits "s is
%     itself not blocked at i-1" (a goal state reaches the goal in 0 steps).
%     Core's choice demands one action per step; this is it.
%   * A blocked cube c is a program part the planner adds on the fly: a rule
%     deriving cubeHolds(c) from the successor state at time 1, a free switch
%     `{ cubeOn(c) }.` and the constraint `:- cubeOn(c), cubeHolds(c).` A query
%     at level i assumes the switch on for every cube of a layer >= i-1 and off
%     for every other one (a switch is a free atom rather than an external
%     because an external is fixed false, and an assumption cannot override
%     that). The goal at time 1 is core's own check(1), switched through
%     query(1).
%
% Scope. Boolean / multi-valued STRIPS with conditional effects and the
% disjunctive layer -- classical planning. Numeric fluents are out, and not
% only because a state at time 0 would need its values as atoms before
% step(1) is grounded: that was tried, with a program grounded per query,
% and it works but a reason can never generalise over a numeric value, so
% the layers grow with every value seen and it loses to deepening by two
% orders of magnitude. Temporal tasks are out with them. Both are refused.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pin the current program part; see the note in core.lp.
#program base.

#defined initialState/2.
#defined pdrInitialState/2.
#defined numVariable/1.
#defined cubeOn/1.
#defined cubeHolds/1.

% The state at time point 0 is free: each boolean variable may hold its true
% value, and holds its false value exactly when it does not. A query fixes
% every one of these choices by assumption.
pdrVariable(V) :- variable(V), not numVariable(V).
{ holds(V, value(V, true), 0) } :- pdrVariable(V).
holds(V, value(V, false), 0) :- pdrVariable(V), not holds(V, value(V, true), 0).

% Doing nothing is a step too (see the header).
action(action(("pdrNoop"))).

% A blocked cube: parts added by the planner as
%   cubeHolds(c) :- holds(V1, value(V1, true), 1), not holds(V2, value(V2, true), 1), ...
%   { cubeOn(c) }.
%   :- cubeOn(c), cubeHolds(c).
% one per learned reason, its switch pinned by assumption in every query.
