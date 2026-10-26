%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential horizon encoding -- TEMPORAL layer.
%
%   owns:     durativeAction/1 snap/3 durationValue/2 overall/3 numOverall/4
%             minSeparation/1
%   requires: core (holds/3, occurs/2, query/1)
%             numeric (numval/numConst/numTerm, for the numOverall rules only --
%             guarded by `#defined`, so they are inert without it)
%
% SMTPlan's happening encoding on top of core's horizon.
%
% (Constraint names in the comments below are the ones used in "A Compilation
% of the Full PDDL+ Language into SMT" / KCL-Planning/SMTPlan's
% EncoderHappening.)
%
% A timestep is a *happening*: happening t sits at absolute time
% time(t) = time(t-1) + delta(t), with time(0) = 0 and delta(t) >= epsilon
% (P3--P4). SMTPlan's happening times are reals and its epsilon is 1/10; clingo
% terms are integers, so the encoder scales every duration in the task by the
% least common denominator and epsilon becomes one scaled unit (minSeparation).
% The discretisation is therefore exact for rational durations, not an
% approximation. Only delta/2 is in the model -- the absolute time of happening
% t is the prefix sum delta(1) + ... + delta(t).
%
% A durative action DA is split into two ordinary actions, its start and end
% *snap* actions, tied back to DA by snap(DA, start|end, A) (H14--H15). At-start
% and at-end conditions/effects are the snaps' own precondition/postcondition
% facts, so every rule in core and numeric applies to them verbatim; this layer
% adds only what couples the two snaps:
%
%   running(DA, t)      SMTPlan's run(a, h): DA is open across the interval that
%                       opens at happening t. True at exactly the happenings
%                       s..e-1 of a start at s and its matching end at e, so
%                       holds(_, s) .. holds(_, e-1) are precisely the states the
%                       open interval (t_s, t_e) spans -- the ones PDDL 2.1 makes
%                       the over-all conditions responsible for.
%   remaining(DA, R, t) SMTPlan's dur(a, h): duration left to run at happening t.
%                       Set when DA starts, decremented by delta at every
%                       happening, and required to be exactly 0 at the end snap
%                       (H9--H10).
%
% Unlike SMTPlan a happening carries exactly one snap action (core's 1{occurs}1
% choice), which makes SMTPlan's mutex constraints between snap actions at the
% same happening vacuous, so they are not encoded. The price is that snap
% actions that must be *simultaneous* cannot be expressed -- they have to be
% sequentialised epsilon apart. Continuous change (PDDL+ processes and events)
% is out of scope: this is the PDDL 2.1 durative-action fragment of SMTPlan.
%
% Cost: the remaining-duration recursion grounds one rule per (remaining value,
% gap) pair per durative action per step, so it is quadratic in the largest
% scaled duration. That is what the encoder's `time_scale` trades against
% resolution -- normalising the durations by their gcd first is what keeps the
% integers small enough for the default of 10 to be affordable.
%
% Every rule here is inert on a task whose facts declare no durativeAction, so
% loading this layer on a classical task is a waste of parse time and nothing
% more.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pin the current program part; see the note in core.lp.
#program base.

#defined durativeAction/1.
#defined snap/3.
% Derived here, but a task grounds check(0) before any step(t) part exists to
% define them; declaring them keeps gringo from reporting that.
#defined running/2.
#defined delta/2.
#defined durationValue/2.
#defined overall/3.
#defined numOverall/4.
#defined minSeparation/1.

% Supplied by the numeric layer; the numOverall rules below are inert without
% it (no numConst/numTerm facts exist to fire them).
#defined numval/3.
#defined numConst/2.
#defined numTerm/3.

#show delta/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The happening timeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

temporal :- durativeAction(_).

% Minimum separation between two happenings (SMTPlan's epsilon) in the encoder's
% scaled time unit; one unit unless the task facts override it.
hasMinSeparation :- minSeparation(_).
separation(E) :- minSeparation(E).
separation(1) :- not hasMinSeparation.

% Candidate gaps. No gap ever has to exceed the longest duration in the task: a
% longer one would step past the end of every action open across it, which the
% remaining-duration constraints below reject anyway.
delay(E) :- separation(E).
delay(D) :- durationValue(_, B), separation(E), D = E..B.

#program step(t).

% t_t = t_{t-1} + delta(t), delta(t) >= epsilon (P3--P4).
1 { delta(t, D) : delay(D) } 1 :- temporal.

% Nothing is open across this gap, so no duration equation spans it and the
% happening may be pulled forward to the earliest legal time. Pruning only.
open(t - 1) :- running(DA, t - 1).
:- delta(t, D), separation(E), D != E, not open(t - 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Snap actions and open intervals (H14--H15)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

starts(DA, t) :- occurs(A, t), snap(DA, start, A).
ends(DA, t)   :- occurs(A, t), snap(DA, end, A).

% run(a, h): open from the start snap until the matching end snap fires.
running(DA, t) :- starts(DA, t).
running(DA, t) :- running(DA, t - 1), not ends(DA, t).

% sta -> not run(h-1): a durative action may not overlap itself.
:- starts(DA, t), running(DA, t - 1).
% end -> run(h-1): an end snap has to close an interval that is open.
:- ends(DA, t), not running(DA, t - 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remaining duration (H9--H10)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sta -> dur = D. The choice is SMTPlan's duration *constraint*: a fixed
% duration leaves one candidate, `lo <= ?duration <= hi` leaves lo..hi.
1 { remaining(DA, D, t) : durationValue(DA, D) } 1 :- starts(DA, t).

% dur(h) = dur(h-1) - delta(h). A value of 0 does not propagate: it is consumed
% by the end snap at this very happening, and a restart re-draws the choice.
remaining(DA, R - D, t) :- remaining(DA, R, t - 1), delta(t, D), R >= D.
hasRemaining(DA, t) :- remaining(DA, _, t).

% end -> dur = 0: the end snap fires exactly when the duration has elapsed.
:- ends(DA, t), not remaining(DA, 0, t).
% run <-> dur > 0: an interval that is still open has time left to run, and a
% gap may not step over an open action's remaining duration.
:- running(DA, t), remaining(DA, 0, t).
:- running(DA, t), not hasRemaining(DA, t).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Over-all conditions: run(a, h) -> condition holds at h
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- running(DA, t), overall(DA, Variable, Value), not holds(Variable, Value, t).

% The numeric analogue, judged at t exactly as numPrecondition is at t-1: the
% two sides' difference as one #sum over numval against the constants' gap
% (see numeric.lp). Needs the numeric layer; inert without it.
:- running(DA, t), numOverall(DA, lt, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } >= C2 - C1.
:- running(DA, t), numOverall(DA, le, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } > C2 - C1.
:- running(DA, t), numOverall(DA, eq, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } != C2 - C1.
:- running(DA, t), numOverall(DA, neq, E1, E2), numConst(E1, C1), numConst(E2, C2),
   #sum { X * K, V, 1 : numTerm(E1, K, V), numval(V, X, t) ;
        -(X * K), V, 2 : numTerm(E2, K, V), numval(V, X, t) } = C2 - C1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Goal: no durative action left open (SMTPlan puts !run(a, H-1) in its goal)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program check(t).

:- query(t), running(DA, t).
