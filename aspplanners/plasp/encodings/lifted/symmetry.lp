%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lifted encoding -- SYMMETRY layer (see aspplanners.plasp.symmetry).
%
%   owns:     symClass/3
%   requires: core (arg/4)
%
% seq's object symmetry breaking, with the objects an action mentions read off
% the chosen arguments: arg(S, I, O, t) is the step's schema S taking object O
% as its I-th argument, so no mentions/2 rules are needed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

#defined symClass/3.
#defined arg/4.

symObject(O) :- symClass(_, _, O).

#program step(t).

used(O, t) :- arg(_, _, O, t), symObject(O).
used(O, t) :- used(O, t - 1).
:- symClass(C, I, O1), symClass(C, I + 1, O2), used(O2, t), not used(O1, t).
