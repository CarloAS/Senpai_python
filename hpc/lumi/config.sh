# Shared settings for the LUMI scripts (sourced by push.sh, pull.sh, setup_env.sh, submit.sh, reconstruct.sbatch).
# Fill in PROJECT and LUMI_USER once.
PROJECT=project_465002965                     # your LUMI project (sbatch --account)
LUMI_USER=sartoric                    # login name on lumi.csc.fi (used by push/pull from the Mac)
LUMI_HOST=lumi.csc.fi
SSH_KEY=$HOME/.ssh/id_lumi                     # key used by push.sh / pull.sh (rsync over ssh -i); Mac side only
export RSYNC_RSH="ssh -i $SSH_KEY"

CODE=/project/$PROJECT/$LUMI_USER/Senpai_python   # git clone of this repo
ENV=/project/$PROJECT/$LUMI_USER/envs/senpai      # container-wrapped Python env (setup_env.sh)
WORK=/scratch/$PROJECT/$LUMI_USER/senpai          # run folder: data/, logs/ (scratch is purged: the Mac keeps the master copy)
