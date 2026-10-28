%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- SYMMETRY layer (see aspplanners.plasp.symmetry).
%
%   owns:     symClass/3 mentions/2
%   requires: core (occurs/2)
%
% Object symmetry breaking. symClass(C, I, O) lists the interchangeable objects
% of class C in clingo's term order; mentions(A, O) says action A names object
% O. Object I+1 may not be used (mentioned by an action that occurs) before
% object I. used/2 is monotone, so "first use of I <= first use of I+1" is
% "used(I+1, t) implies used(I, t)" at every t. A lex-leader constraint: the
% term-smallest plan of every relabelling class is kept, so a horizon that had
% a plan still has one.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

#defined symClass/3.
#defined mentions/2.
#defined occurs/2.

symObject(O) :- symClass(_, _, O).

#program step(t).

used(O, t) :- occurs(A, t), mentions(A, O), symObject(O).
used(O, t) :- used(O, t - 1).
:- symClass(C, I, O1), symClass(C, I + 1, O2), used(O2, t), not used(O1, t).
