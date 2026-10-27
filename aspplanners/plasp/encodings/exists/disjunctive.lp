%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exists-step horizon encoding -- DISJUNCTIVE layer.
%
% A verbatim copy of seq/disjunctive.lp: the `exists` family keeps its own files
% (see exists/core.lp). Its rules are per action and per step, so they read the
% same under a set of actions per step as under one. The exists-step propagator
% reads this layer's facts to know what each action reads and writes.
%
%   owns:     orGroup/2 orDisjunct/5 orDisjunctId/3 orDisjunctNum/6 orDisjunctSub/4
%             goalOrGroup/1 goalOrDisjunct/4 goalOrDisjunctId/2 goalOrDisjunctNum/5
%             goalOrDisjunctSub/3
%   requires: core (holds/3, occurs/2, query/1)
%             numeric, but only for the orDisjunctNum/goalOrDisjunctNum rules --
%             they are guarded by `#defined` numval/numConst/numTerm and inert without it,
%             so a task with disjunctions but no arithmetic needs core only.
%
% This layer also covers `exists` and nested/quantified disjunctions, which the
% encoder emits into the same vocabulary; see the group/disjunct discussion below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pin the current program part; see the note in core.lp.
#program base.

#defined orGroup/2.
#defined orDisjunct/5.
#defined orDisjunctId/3.
#defined orDisjunctNum/6.
#defined orDisjunctSub/4.
#defined goalOrGroup/1.
#defined goalOrDisjunct/4.
#defined goalOrDisjunctId/2.
#defined goalOrDisjunctNum/5.
#defined goalOrDisjunctSub/3.

% Supplied by the numeric layer; the *Num rules below are inert without it
% (no numConst/numTerm facts exist to fire them).
#defined numval/3.
#defined numConst/2.
#defined numTerm/3.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Disjunctive preconditions (and disjunctive conditions of conditional effects)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

% precondition/3 facts are read conjunctively, which is what lets a `forall` be
% emitted as one rule with its variable ranged by the grounder. A disjunction
% cannot be stated that way, so it gets its own vocabulary: an action carries one
% orGroup/2 per disjunctive condition, each holding orDisjunct/5 literals grouped
% by disjunct. At least one disjunct has to hold, and a disjunct holds when every
% literal in it does. An `exists` is the same thing with the disjuncts indexed by
% the quantified variable's binding, so the grounder enumerates them. A disjunct's
% numeric literals are orDisjunctNum/6 (checked as a difference #sum over numval, like a numeric
% precondition), and orDisjunctId/3 declares the disjunct itself so one made only
% of numeric literals is enumerated as well.
%
% A `forall` over a disjunction carries its variable in the group term, so it is
% one group per binding and the constraint below demands every one of them.
% A disjunction *nested* in a disjunct is a group of its own, declared without an
% orGroup/2 fact (nothing forces it by itself) and required by the orDisjunctSub/4
% atom of the disjunct that owns it -- which is how the flat vocabulary states an
% arbitrarily nested condition. The negative loop between orGroupHolds/3 and
% orDisjunctFails/4 is only apparent: a subgroup is strictly nested inside its
% parent, so no ground instance ever depends on itself.
%
% The `orGroup(Effect, G)` groups of a conditional effect are read by core's
% caused/3 rule rather than by the constraint here, which is guarded by occurs/2.
orDisjunctFails(A, G, D, t) :- orDisjunct(A, G, D, Variable, Value), not holds(Variable, Value, t - 1).
orDisjunctFails(A, G, D, t) :- orDisjunctSub(A, G, D, SubGroup), not orGroupHolds(A, SubGroup, t).
% A disjunct's numeric literal is judged like a numeric precondition
% (numeric.lp): the two sides' difference as one #sum over numval against the
% constants' gap, one ground rule per literal and step -- never a join of two
% reified values, whose instances multiply with the reachable value range.
orDisjunctFails(A, G, D, t) :- orDisjunctNum(A, G, D, lt, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } >= C2 - C1.
orDisjunctFails(A, G, D, t) :- orDisjunctNum(A, G, D, le, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } > C2 - C1.
orDisjunctFails(A, G, D, t) :- orDisjunctNum(A, G, D, eq, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } != C2 - C1.
orDisjunctFails(A, G, D, t) :- orDisjunctNum(A, G, D, neq, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } = C2 - C1.
orGroupHolds(A, G, t) :- orDisjunctId(A, G, D), not orDisjunctFails(A, G, D, t).
:- occurs(A, t), orGroup(A, G), not orGroupHolds(A, G, t).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Disjunctive goals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

% Read exactly as disjunctive preconditions are, but against the state at the
% goal step rather than the one before the action.
goalDisjunctFails(G, D, t) :- goalOrDisjunct(G, D, Variable, Value), not holds(Variable, Value, t).
goalDisjunctFails(G, D, t) :- goalOrDisjunctSub(G, D, SubGroup), not goalOrGroupHolds(SubGroup, t).
goalDisjunctFails(G, D, t) :- goalOrDisjunctNum(G, D, lt, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } >= C2 - C1.
goalDisjunctFails(G, D, t) :- goalOrDisjunctNum(G, D, le, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } > C2 - C1.
goalDisjunctFails(G, D, t) :- goalOrDisjunctNum(G, D, eq, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } != C2 - C1.
goalDisjunctFails(G, D, t) :- goalOrDisjunctNum(G, D, neq, E1, E2), numConst(E1, C1), numConst(E2, C2),
    #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
         -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } = C2 - C1.
goalOrGroupHolds(G, t) :- goalOrDisjunctId(G, D), not goalDisjunctFails(G, D, t).
:- query(t), goalOrGroup(G), not goalOrGroupHolds(G, t).
