import mne
import numpy as np
import matplotlib.pyplot as plt

mne.set_log_level("WARNING")

psg_path = r"C:\Users\cierr\Documents\BME_PYTH\sleep_final\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
hyp_path = r"C:\Users\cierr\Documents\BME_PYTH\sleep_final\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001EC-Hypnogram.edf"

raw = mne.io.read_raw_edf(psg_path, preload=True)
annotations = mne.read_annotations(hyp_path)
raw.set_annotations(annotations)
raw.pick(["EEG Fpz-Cz"])
fs = int(raw.info["sfreq"])

stage_map = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}
stage_labels = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

hyp_data = []

for desc, dur in zip(
        annotations.description,
        annotations.duration):

    if desc not in stage_map:
        continue

    n_epochs = int(dur / 30)

    hyp_data.extend(
        [stage_map[desc]] * n_epochs
    )

hyp_data = np.array(hyp_data)
sleep_epochs = np.where(hyp_data > 0)[0]
buffer_epochs = 10
start_epoch = max(
    0,
    sleep_epochs[0] - buffer_epochs
)

end_epoch = min(
    len(hyp_data),
    sleep_epochs[-1] + buffer_epochs
)

hyp_trimmed = hyp_data[start_epoch:end_epoch]

eeg = raw.get_data()[0]
samples_per_epoch = 30 * fs

stage_example_epochs = {}
for stage in [0, 1, 2, 3, 4]:
    idx = np.where(hyp_data == stage)[0]
    if len(idx) > 0:
        middle_idx = idx[len(idx) // 2]
        stage_example_epochs[stage] = middle_idx

plt.figure(figsize=(15, 5))
plt.step(
    np.arange(len(hyp_trimmed)),
    hyp_trimmed,
    where="post",
    linewidth=2,
    color="purple"
)

marker_colors = {
    0: "black",
    1: "red",
    2: "blue",
    3: "green",
    4: "orange"
}

for stage, epoch in stage_example_epochs.items():
    if start_epoch <= epoch <= end_epoch:
        x = epoch - start_epoch
        plt.scatter(
            x,
            stage,
            color=marker_colors[stage],
            s=80,
            zorder=5
        )
        plt.text(
            x,
            stage + 0.15,
            stage_labels[stage],
            fontsize=9,
            ha="center"
        )

plt.gca().invert_yaxis()

plt.yticks(
    [0, 1, 2, 3, 4],
    ["Wake", "N1", "N2", "N3", "REM"]
)

plt.xlabel("30-Second Epochs")
plt.ylabel("Sleep Stage")
plt.title("Sleep Hypnogram (Trimmed Around Sleep Period)")
plt.grid(True)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    5,
    1,
    figsize=(15, 10)
)

for i, stage in enumerate([0, 1, 2, 3, 4]):
    if stage not in stage_example_epochs:
        continue
    epoch = stage_example_epochs[stage]
    start_sample = epoch * samples_per_epoch
    end_sample = start_sample + samples_per_epoch
    segment = eeg[start_sample:end_sample]
    time = np.arange(len(segment)) / fs
    axes[i].plot(
        time,
        segment,
        linewidth=1
    )
    axes[i].set_title(
        f"{stage_labels[stage]} Example EEG"
    )
    axes[i].set_ylabel("uV")
    axes[i].grid(True)

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.show()

raw.compute_psd(fmax=35).plot()
plt.show(block=True)

stage_minutes = {}
for desc, dur in zip(
        annotations.description,
        annotations.duration):
    if desc not in stage_map:
        continue
    stage = stage_map[desc]
    stage_minutes[stage] = (
        stage_minutes.get(stage, 0)
        + dur / 60
    )

plt.figure(figsize=(8, 5))
stages = sorted(stage_minutes.keys())
durations = [
    stage_minutes[s]
    for s in stages
]

labels = [
    stage_labels[s]
    for s in stages
]

plt.bar(
    labels,
    durations,
    color="steelblue"
)

plt.ylabel("Minutes")
plt.title("Time Spent in Each Sleep Stage")

for i, v in enumerate(durations):
    plt.text(
        i,
        v + 5,
        f"{v:.1f}",
        ha="center"
    )

plt.tight_layout()
plt.show()
