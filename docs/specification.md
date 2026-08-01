# Specification

A condensed version of the project brief. The original is longer and carries
per-stage notes and internal planning; what follows is everything that matters
to someone reading the code.

## 1. What the system is

NES Player is a research system in which an autonomous agent learns to play
Nintendo Entertainment System games through an interface as close to human
perception and control as practical.

The agent receives:

- the video stream of the game;
- the audio stream of the game;
- the history of its own actions;
- the internal state of its own model;
- optionally, a small number of human demonstrations.

It acts through a standard NES controller: `A`, `B`, `Start`, `Select`, `Up`,
`Down`, `Left`, `Right`.

During normal training and during final evaluation the agent must not receive
object coordinates, health, score, level state, a collision map or any other
semantic data directly from emulator memory.

## 2. Research goal

Build an agent that learns transferable game representations and skills rather
than one particular game. It should be able to:

- find the object it controls;
- segment and follow visual entities;
- link game sounds to events;
- establish a causal link between buttons and changes on screen;
- estimate local dynamics;
- recognise threats, goals and signs of progress;
- learn from human and TAS demonstrations;
- adapt to a new game;
- plan inside its own world model;
- carry skills across games of different genres.

## 3. Hypothesis

An agent pretrained on a set of dissimilar NES games should learn a new game
faster than an agent trained from scratch. The transfer should come not from
memorising particular sprites but from general concepts: the controlled object,
an enemy, a projectile, a platform, an obstacle, a passage, the screen edge,
loss of the controlled object, a HUD change, the start and end of an episode, a
visual or audible sign of progress, a repeatable motor sequence, and the causal
link between an action and its result.

## 4. Principles

**A human-like interface.** The policy sees what a player sees and hears what a
player hears. Nothing is fed in through a side channel.

**Audio is a first-class input, not decoration.** A hit, damage, a pickup or a
scene change is often audible before it is visible, and sometimes happens off
screen entirely. Any claim that audio helps must be backed by an ablation.

**No hidden telemetry in the policy.** Emulator memory is available to the
training loop and to evaluation as ground truth — to check whether what the
agent read off the screen was correct — and never to the policy itself.

**Reproducibility.** A fixed seed and a fixed movie must produce byte-identical
frames and audio. Regression tests hold golden hashes of a reference run; they
are what caught a third-party core silently replacing the default one.

## 5. Scope of the MVP

In scope: the emulator harness, a dataset builder, behavioural cloning with and
without audio, discovery of the controlled object, a world model with a
planner, and measured transfer between games.

Out of scope for the first version: finishing any game, real-time performance
on hardware other than a laptop, analogue controls, other consoles, and
language-conditioned goals.

## 6. Acceptance criteria

The MVP is accepted when all of the following hold. All ten are met; see
[experiments.md](experiments.md) for the numbers behind each.

1. The emulator runs headless.
2. Video, audio and actions are recorded in sync.
3. At least one TAS movie replays to the end without desynchronising.
4. Datasets build reproducibly.
5. Behavioural cloning beats a random policy.
6. Adding audio measurably improves at least one task.
7. The system identifies the controlled object by itself in at least three games.
8. The world model predicts short-term dynamics better than a baseline that
   ignores actions.
9. The planner improves short-term behaviour over the bare policy.
10. A pretrained agent learns at least one new game faster than one from scratch.
11. Every experiment is reproducible from a script.
12. Reference runs automatically produce a video with the debug overlay.

## 7. Metrics

**Perception** — object tracking consistency, controlled-object identification
accuracy, event boundary F1, audio event clustering quality, audio↔video
retrieval accuracy.

**World model** — latent prediction error, object trajectory error, event
prediction accuracy, uncertainty calibration, prediction horizon before
divergence.

**Policy** — action accuracy, sequence accuracy, episode return, survival time,
progress score, completion rate, recovery rate after a deviation, action
entropy, repeated-failure count.

**Transfer** — time to the first meaningful action, actions needed to identify
the controls, progress after a fixed budget, improvement over training from
scratch, zero-shot score, few-shot sample efficiency.

**System** — emulator frames per second, environment steps per second, training
samples per second, accelerator and CPU utilisation, memory use, thermal
throttling, data loading latency.

An accuracy is never reported without its baseline next to it. A policy that is
90% accurate on a game where one button covers 88% of frames has learned
nothing.

## 8. Roadmap

| Stage | State |
|---|---|
| 0. Repository and infrastructure | done |
| 1. Emulator harness | done |
| 2. Dataset builder | done |
| 3. Behavioural cloning baseline | done |
| 4. Audio | done |
| 5. Object discovery | partial — motion tracking works, neural slots did not |
| 6. World model | done |
| 7. Planner | done |
| 8. Multi-game pretraining | done |
| 9. Online adaptation during play | not started |
| 10. Skill library | not started |

## 9. After the MVP

Possible extensions: other consoles, analogue controls, language-conditioned
goals, learning from ordinary video, a vision-language model for describing
events, hierarchical planning, automatic curricula, self-play, distributed
rollout workers.

## 10. The criterion that actually matters

The project is a research success not when one network finishes one game, but
when it can be shown convincingly that:

> representations learned on some games reduce the amount of experience needed
> to learn others.

That is what separates this from a reinforcement-learning bot that has
memorised one fixed environment.
