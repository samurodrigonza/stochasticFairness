This folder is the default output location for `run_experiments.py` runs.
The final scarcity-0.75, 150-scenario, MIPGap=1e-6 snapshot is tracked here
and also preserved in the repository-root `results.zip`, by explicit request.

New generated paths remain git-ignored, but rerunning the pipeline can modify
the tracked snapshot and overwrite per-solve logs. Preserve results before
rerunning and stage output changes deliberately. See `docs/agent_handoff.md`
and `docs/dossier.md` section 15.11 for provenance and interpretation.
