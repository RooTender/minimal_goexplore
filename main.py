import os
from collections import defaultdict, namedtuple
from threading import Thread, Lock
from time import sleep
import numpy as np
import cv2
import gymnasium as gym

def cellfn(frame):
    cell = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cell = cv2.resize(cell, (11, 8), interpolation = cv2.INTER_AREA)
    cell = cell // 32
    return cell

def hashfn(cell):
    return hash(cell.tobytes())

class Weights:
    times_chosen = 0.1
    times_chosen_since_new = 0
    times_seen = 0.3

class Powers:
    times_chosen = 0.5
    times_chosen_since_new = 0.5
    times_seen = 0.5

class Cell(object):
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key != 'score' and hasattr(self, 'times_seen'):
            self.score = self.cellscore()

    def cntscore(self, a):
        w = getattr(Weights, a)
        p = getattr(Powers, a)
        v = getattr(self, a)
        return w / (v + e1) ** p + e2

    def cellscore(self):
        return self.cntscore('times_chosen')           +\
               self.cntscore('times_chosen_since_new') +\
               self.cntscore('times_seen')             +\
               1

    def visit(self):
        self.times_seen += 1
        return self.times_seen == 1

    def choose(self):
        self.times_chosen += 1
        self.times_chosen_since_new += 1
        return self.ram, self.reward, self.trajectory

archive = defaultdict(lambda: Cell())
highscore = 0
frames = 0
iterations = 0

highscore_lock = Lock()
new_cell = np.zeros((1, 1, 3))

highscore_updated = False
replay_frames = []

e1 = 0.001
e2 = 0.00001

def replay_highscore_frames(score, highscore_frames):
    cv2.namedWindow(f"Best Score {score}", cv2.WINDOW_NORMAL)

    for _ in range(3):
        for frame in highscore_frames:
            cv2.imshow(f"Best Score {score}", frame)
            cv2.waitKey(30)

        cv2.waitKey(3000)

    cv2.destroyWindow(f"Best Score {score}")

def explore():
    global highscore, highscore_updated, replay_frames, frames, iterations, new_cell, archive

    env = gym.make("MontezumaRevengeDeterministic-v4")
    frame = env.reset()
    score = 0
    action = 0
    trajectory = []
    my_iterations = 0

    local_frames = []

    while True:
        found_new_cell = False
        episode_length = 0

        for _ in range(100):
            if np.random.random() > 0.95:
                action = env.action_space.sample()

            frame, reward, terminal, terminal, info = env.step(action)
            score += reward
            terminal |= info['lives'] < 6

            trajectory.append(action)
            episode_length += 4

            if score > highscore:
                with highscore_lock:
                    if score > highscore:
                        highscore = score
                        replay_frames = local_frames.copy()
                        highscore_updated = True

            local_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Store frame for replay

            if terminal:
                frames += episode_length
                break
            else:
                cell = cellfn(frame)
                cellhash = hashfn(cell)
                cell = archive[cellhash]
                first_visit = cell.visit()
                if first_visit or score > cell.reward or score == cell.reward and len(trajectory) < len(cell.trajectory):
                    cell.ram = env.unwrapped.clone_state(include_rng=True)
                    cell.reward = score
                    cell.trajectory = trajectory.copy()
                    cell.times_chosen = 0
                    cell.times_chosen_since_new = 0
                    new_cell = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    found_new_cell = True

        if found_new_cell and my_iterations > 0:
            restore_cell.times_chosen_since_new = 0

        scores = np.array([cell.score for cell in archive.values()])
        hashes = [cellhash for cellhash in archive.keys()]
        probs = scores / scores.sum()
        restore = np.random.choice(hashes, p = probs)
        restore_cell = archive[restore]
        ram, score, trajectory = restore_cell.choose()
        env.reset()
        env.unwrapped.restore_state(ram)
        my_iterations += 1
        iterations += 1

threads = [Thread(target = explore) for _ in range(os.cpu_count() - 1)]

for thread in threads:
    thread.start()

while True:
    print ("Iterations: %d, Cells: %d, Frames: %d, Max Reward: %d" % (iterations, len(archive), frames, highscore))

    if highscore_updated:
        replay_highscore_frames(highscore, replay_frames)
        highscore_updated = False
