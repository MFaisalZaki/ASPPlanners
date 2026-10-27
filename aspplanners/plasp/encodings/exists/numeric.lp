%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exists-step horizon encoding -- NUMERIC layer.
%
% A verbatim copy of seq/numeric.lp: the `exists` family keeps its own files
% (see exists/core.lp). Its rules are per action and per step, so they read the
% same under a set of actions per step as under one. The exists-step propagator
% reads this layer's facts to know what each action reads and writes.
%
%   owns:     numVariable/1 numConst/2 numTerm/3 numPrecondition/4 numEffect/3
%             numAssign/3 numEffectExpr/3 numAssignExpr/3 numGoal/3
%             numCondEffect/4 numCondEffectExpr/4 numCondAssignExpr/4
%   requires: core (holds/3, occurs/2, the caused/modified inertia)
%             disjunctive, but only where a conditional numeric effect carries a
%             disjunctive (or numeric) condition -- read through `#defined`
%             orGroup/orGroupHolds, so the rules are inert without it
%
% Numeric fluents ride on core's holds/3 chain -- a numeric variable's value is
% carried as `holds(V, value(V, X), t)`, so core's inertia rule propagates it and
% this layer adds no frame axiom of its own. numval/3 projects that atom to its
% integer and numValue/3 evaluates linear expressions over it.
%
% numval/3 (and the numTerm/numConst side facts) is the interface other layers
% read: the disjunctive layer's numeric literals and the temporal layer's
% over-all comparisons state each comparison as a difference #sum over numval,
% exactly as the precondition and goal comparisons below do, and are inert
% without this layer because no numConst/numTerm facts exist to fire them.
% numValue/3 -- the expression's value REIFIED into an atom -- survives only
% where the value must become part of a derived term: the expression effects
% (numEffectExpr/numAssignExpr), which bind it as the delta or the assigned
% value. Everything else deliberately avoids the atom form: an assignment
% aggregate grounds one candidate rule instance per reachable value, each
% carrying its whole element list, and a comparison joining two such atoms
% grounds one instance per VALUE PAIR -- both blow up as reachable ranges
% widen with the horizon, which is the dominant grounding cost on numeric
% tasks. A difference #sum grounds once per guard and step instead.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Every layer file opens by pinning the current program part, so that loading the
% layers as separate files and concatenating them into one program (what
% PLASPPlanner.lp_program does) are the same program. Without it the statements
% above a file's first `#program` would inherit the part the previous file
% happened to end in.
#program base.

#defined numGoal/3.
#defined numVariable/1.
#defined numConst/2.
#defined numTerm/3.
#defined numPrecondition/4.
#defined numEffect/3.
#defined numAssign/3.
#defined numEffectExpr/3.
#defined numAssignExpr/3.
#defined numCondEffect/4.
#defined numCondEffectExpr/4.
#defined numCondAssignExpr/4.

% Read by the conditional-effect rules below, which are how a numeric effect
% gets a condition. `precondition/3` is core's; the orGroup pair is the
% disjunctive layer's, and is also where a *numeric* condition on a conditional
% effect lives -- the encoder states it as a one-disjunct group, so it is judged
% by that layer's difference #sum rather than needing a rule of its own here.
#defined precondition/3.
#defined orGroup/2.
#defined orGroupHolds/3.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% numval projects a numeric variable's holds atom to its integer.
numval(V, X, 0) :- holds(V, value(V, X), 0), numVariable(V).

% A side of a numeric comparison is the linear expression `K1*V1 + ... + C`.
% numConst/2 states its constant and declares it (so a constant-only side, whose
% sum is empty, still gets a value); one numTerm/3 states each K*V. Keeping the
% terms in their own facts is what lets a side hold any number of fluents
% without the comparison's arity depending on how many -- `(+ (x ?b) (y ?b))`
% and `(* 2 (x ?f))` are ordinary expressions here, not special cases.
%
% numValue is derived ONLY for the expressions the effect rules read (see the
% header): those must bind the value as a term. Comparisons never go through
% it, so a task without expression effects grounds no numValue at all.
numValueNeeded(E) :- numEffectExpr(_, _, E).
numValueNeeded(E) :- numAssignExpr(_, _, E).
numValueNeeded(E) :- numCondEffectExpr(_, _, _, E).
numValueNeeded(E) :- numCondAssignExpr(_, _, _, E).
numValue(E, S + C, 0) :- numValueNeeded(E), numConst(E, C), S = #sum { X * K, V : numTerm(E, K, V), numval(V, X, 0) }.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preconditions, effects, and the value of every expression at t
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program step(t).

% Numeric preconditions: linear comparisons of the two sides at t-1, stated as
% ONE #sum over the difference of their term sums against the constants' gap:
% `S1 >= S2` is `T1 + C1 >= T2 + C2` is `T1 - T2 >= C2 - C1`. One ground rule
% per guard and step, elements linear in the sides' fluents -- where a join of
% two reified values grounds one instance per candidate value PAIR. The side
% tags (1/2) keep an element of E1 distinct from the same fluent's element of
% E2, so a fluent shared by both sides is summed once per side, and comparing
% an expression against itself sums to zero as it should. The positive
% numConst atoms keep the original vacuousness: a comparison whose side was
% never declared constrains nothing.
:- occurs(A, t), numPrecondition(A, lt, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } >= C2 - C1.
:- occurs(A, t), numPrecondition(A, le, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } > C2 - C1.
:- occurs(A, t), numPrecondition(A, eq, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } != C2 - C1.
:- occurs(A, t), numPrecondition(A, neq, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t - 1) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t - 1) } = C2 - C1.

% Numeric effects: constant-delta updates and constant assignments. They feed
% core's caused/3, so core's inertia and modified/2 handle them like any effect.
caused(V, value(V, X + D), t) :- occurs(A, t), numEffect(A, V, D), holds(V, value(V, X), t - 1).
caused(V, value(V, K), t) :- occurs(A, t), numAssign(A, V, K).

% Effects whose delta or assigned value is itself a linear expression over the
% state: the expression is evaluated at t-1 through numValue/3, exactly as a
% precondition's sides are, and applied at t. Kept apart from the constant
% forms so a task without them grounds exactly what it did before.
caused(V, value(V, X + D), t) :- occurs(A, t), numEffectExpr(A, V, E), holds(V, value(V, X), t - 1), numValue(E, D, t - 1).
caused(V, value(V, D), t) :- occurs(A, t), numAssignExpr(A, V, E), numValue(E, D, t - 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conditional numeric effects
%
% A conditional *boolean* effect rides on core's caused/3, which states the
% variable's new value outright -- and so does a conditional numeric assignment
% of a constant. The other numeric shapes cannot: a delta's new value is the old
% one plus something, and PDDL's reading of two deltas that both fire is that
% they *add*. Two caused/3 atoms would be two states rather than one sum, so the
% deltas are collected first and applied once.
%
% An effect fires under exactly the conditions core's caused/3 fires one: every
% precondition/3 atom of its effect term holds at the step before, and so does
% every disjunctive group the term carries. A numeric condition is one of those
% groups (a single disjunct holding a single comparison), which is what lets it
% be judged by the disjunctive layer's difference #sum instead of a rule here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numCondTerm(A, Eff) :- numCondEffect(A, Eff, _, _).
numCondTerm(A, Eff) :- numCondEffectExpr(A, Eff, _, _).
numCondTerm(A, Eff) :- numCondAssignExpr(A, Eff, _, _).

numEffectFires(A, Eff, t) :- occurs(A, t), numCondTerm(A, Eff),
    holds(VariablePre, ValuePre, t - 1) : precondition(Eff, VariablePre, ValuePre);
    orGroupHolds(Eff, G, t) : orGroup(Eff, G).

% One element per firing delta, keyed by its effect term so that two effects
% with the same delta both count -- and so that a `forall` conditional effect,
% whose term carries the binding, contributes once per binding.
numDelta(V, Eff, D, t) :- numCondEffect(A, Eff, V, D), numEffectFires(A, Eff, t).
numDelta(V, Eff, D, t) :- numCondEffectExpr(A, Eff, V, E), numEffectFires(A, Eff, t),
                          numValue(E, D, t - 1).
numDeltaFired(V, t) :- numDelta(V, _, _, t).
caused(V, value(V, X + S), t) :- numDeltaFired(V, t), holds(V, value(V, X), t - 1),
                                 S = #sum { D, Eff : numDelta(V, Eff, D, t) }.

% A conditional assignment whose value reads the state, evaluated at t-1 like
% any other expression. The constant case never reaches here: it rides on
% core's postcondition/caused, exactly as a conditional boolean effect does.
caused(V, value(V, D), t) :- numCondAssignExpr(A, Eff, V, E), numEffectFires(A, Eff, t),
                             numValue(E, D, t - 1).

numval(V, X, t) :- holds(V, value(V, X), t), numVariable(V).
numValue(E, S + C, t) :- numValueNeeded(E), numConst(E, C), S = #sum { X * K, V : numTerm(E, K, V), numval(V, X, t) }.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Numeric goals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

% Linear comparisons at the goal step, stated as difference #sums over
% numval(_, _, t) exactly as numeric preconditions are at t-1.
:- query(t), numGoal(lt, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } >= C2 - C1.
:- query(t), numGoal(le, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } > C2 - C1.
:- query(t), numGoal(eq, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } != C2 - C1.
:- query(t), numGoal(neq, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } = C2 - C1.
