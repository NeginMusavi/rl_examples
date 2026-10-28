# Policy demonstrations

## Export

The notebook video cell calls `simulate_runs` with `video_dir`. Each selected
checkpoint is evaluated deterministically on the specified scenario seeds.

Each episode produces:

| File | Content |
|---|---|
| `.mp4` | H.264 video |
| `.png` | Final-frame preview |
| `.json` | Algorithm, environment settings, training seed, checkpoint steps, scenario, and outcome |

Navigation videos show one frame per high-level decision at 2 fps, plus the
initial pose. An action spans at most 0.5 simulated seconds; early terminal actions
can end sooner. The video is a decision-level view, not a rendering of every
integration step. LunarLander records the initial frame and each physics step at
50 fps. Encoding streams frames to disk.

## README presentation

Copy selected videos and previews from `outputs/<run>/videos/` to `docs/media/`.
Use descriptive names such as `navigation_sac.mp4` and `navigation_sac.png`.
Only this documentation media directory is exempted from the MP4 Git ignore rule.

```markdown
[![SAC navigation, margin = 0.1 m](docs/media/navigation_sac.png)](docs/media/navigation_sac.mp4)
```

This creates a clickable image linked to the video; it does not assume an inline
MP4 player. A short animated GIF can be used as an alternative preview.
GitHub also supports video attachments in issues and pull requests. For larger
collections, host videos externally and link the preview images to them.

Describe whether each clip is representative, a failure case, or a selected
successful example. Keep demonstrations separate from aggregate evaluation results.
Smoke-test clips are execution checks and should not be presented as trained performance.

References: [GitHub relative links and images](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
and [supported attachments](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files).
