%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- SYMMETRY layer (opt-in, see aspplanners.plasp.symmetry).
%
%   owns:     symClass/3 mentions/2 symOrder/0
%   requires: core (action/1, precondition/3, postcondition/4, occurs/2)
%
% Lex-leader constraints over two symmetries of the plan space. Both keep the
% term-smallest plan of every symmetry class, so a horizon that had a plan still
% has one.
%
% 1. Object symmetries. symClass(C, I, O) lists the interchangeable objects of
%    class C in clingo's term order. Object I+1 may not be used (mentioned by an
%    action that occurs) before object I. used/2 is monotone, so "first use of
%    I <= first use of I+1" is "used(I+1, t) implies used(I, t)" at every t.
%
% 2. Plan permutation symmetries, enabled by the symOrder fact. Two adjacent
%    actions neither of which writes a variable the other reads or writes can
%    be swapped, so they have to appear in ascending term order. The relation
%    is at the level of variables, computed once from the action facts; an
%    action with a shape this layer does not analyse (a numeric conditional
%    effect, a numeric disjunct) is opaque and interferes with everything.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

#defined symClass/3.
#defined mentions/2.
#defined symOrder/0.
#defined occurs/2.
#defined action/1.
#defined precondition/3.
#defined postcondition/4.
#defined orDisjunct/5.
#defined orDisjunctNum/6.
#defined numPrecondition/4.
#defined numEffect/3.
#defined numAssign/3.
#defined numEffectExpr/3.
#defined numAssignExpr/3.
#defined numCondEffect/4.
#defined numCondEffectExpr/4.
#defined numCondAssignExpr/4.
#defined numTerm/3.

symObject(O) :- symClass(_, _, O).

% What an action reads and writes, by variable.
writes(A, V) :- postcondition(A, _, V, _).
writes(A, V) :- numEffect(A, V, _).
writes(A, V) :- numAssign(A, V, _).
writes(A, V) :- numEffectExpr(A, V, _).
writes(A, V) :- numAssignExpr(A, V, _).
reads(A, V)  :- precondition(A, V, _).
reads(A, V)  :- postcondition(A, E, _, _), precondition(E, V, _).
reads(A, V)  :- postcondition(A, E, _, _), orDisjunct(E, _, _, V, _).
reads(A, V)  :- orDisjunct(A, _, _, V, _).
reads(A, V)  :- numPrecondition(A, _, E1, _), numTerm(E1, _, V).
reads(A, V)  :- numPrecondition(A, _, _, E2), numTerm(E2, _, V).
reads(A, V)  :- numEffectExpr(A, _, E), numTerm(E, _, V).
reads(A, V)  :- numAssignExpr(A, _, E), numTerm(E, _, V).
opaque(A)    :- numCondEffect(A, _, _, _).
opaque(A)    :- numCondEffectExpr(A, _, _, _).
opaque(A)    :- numCondAssignExpr(A, _, _, _).
opaque(A)    :- orDisjunctNum(A, _, _, _, _, _).
opaque(A)    :- postcondition(A, E, _, _), orDisjunctNum(E, _, _, _, _, _).

interferes(A, B) :- writes(A, V), reads(B, V).
interferes(A, B) :- writes(A, V), writes(B, V).
interferes(A, B) :- opaque(A), action(B).
interferes(A, B) :- opaque(B), action(A).

% The ordered pairs a sequential plan may not list adjacently.
swappable(A, B) :- symOrder, action(A), action(B), A > B,
                   not interferes(A, B), not interferes(B, A).

#program step(t).

used(O, t) :- occurs(A, t), mentions(A, O), symObject(O).
used(O, t) :- used(O, t - 1).
:- symClass(C, I, O1), symClass(C, I + 1, O2), used(O2, t), not used(O1, t).

:- occurs(A, t - 1), occurs(B, t), swappable(A, B).
