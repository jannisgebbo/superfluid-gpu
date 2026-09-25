#!/usr/bin/env python3
import json
import math
import os
import random
import subprocess
import sys
import uuid

import dstack

random.seed()

data = {
    # lattice dimension
    "NX": 32,

    # Time stepping
    "finaltime": 10,
    "initialtime": 0,
    "deltat": 0.5,

    # Action of O4 mdoel parameters
    "H": 0.003,
    "chi": 5.0,

    # Transport Coefficients
    "gamma": 0.25,
    "diffusion": 0.25 / 3.0,
    # evolverType: the split stepper for the superfluid sector.
    #   'A' = ideal (Verlet) substep, 'B' = explicit diffusion substep.
    "evolverType": "SuperSplitStep",
    "SuperSplitStep": {
        "step_sequence": "ABBBBBB",
    },
    "seed": 122335456,
    "outputfiletag": "grun",
    "saveFrequency": 3,
    # Options are [ "default", "randomspins-gaussiancharges", "loadfile",
    # "gaussians", "spinwaves" ]. "default" draws random phi on the sphere
    # with zero charges. "randomspins-gaussiancharges" adds gaussian charges
    # of zero total (SuperQuench's "randomspins"; for regression runs).
    "initialization": "default",
    # For running multi-events
    "eventmode": False,
    "nevents": 1,
    # parameters for superfluid events (the code is always in superfluid mode)
    "f2_constant": 5.0,
}

# This is a flag to toggle input checking
CheckInputs = True

# Check that the setup is sensible


def checkinputs():
    if not CheckInputs:
        return
    if data["chi"] != 5.0:
        raise SystemExit("Chi should be five")
    if data["f2_constant"] != 5.0:
        raise SystemExit("f2 constant should be five")
    if data["evolverType"] != "SuperSplitStep":
        raise SystemExit('evolverType must be "SuperSplitStep"')


# dump the data into a .json file
def datatojson():
    with open(data["outputfiletag"] + ".json", "w") as outfile:
        json.dump(data, outfile, indent=4)


def getdefault_filename(tag):
    tag = data["outputfiletag"]
    name = "%s_N%03d_h%06d_c%05d" % (
        tag,
        data["NX"],
        round(1000000 * data["H"]),
        round(100 * data["chi"]),
    )
    return name


# Find the program looking in the environment variable for the path
def find_program(program_name="SuperPions.exe"):
    path = os.environ.get("MODELGEXEPATH")
    if path is None:
        print("Unable to find the path MODELGEXEPATH")
        sys.exit(1)

    abspath = os.path.join(path, program_name)
    if os.path.exists(abspath):
        print("Found the executable {}".format(abspath))
    else:
        print("Unable to find the executable {}".format(abspath))
    return abspath


#########################################################################
# Runs on perlmutter
#########################################################################
def prlmrun(
    time=2,
    debug=False,
    dry_run=True,
    moreopts=["-log_view"],
    seed=None,
    nnodes=1,
    nodeid=False,
    interactive=False,
    gpu=False,
):
    """Write (and unless dry_run, submit) a Perlmutter job script.

    gpu=False: CPU nodes (account m3722), 128 ranks per node, host-loop
    build from $MODELGEXEPATH.
    gpu=True: GPU nodes (account m3722_g), 4 ranks per node, one A100 each,
    Kokkos vectors. Uses $MODELGPATH/build-gpu and sources
    setupprlm-gpu.sh inside the job, whatever environment was loaded at
    submission.
    """
    if gpu:
        modelgpath = os.environ.get("MODELGPATH")
        if modelgpath is None:
            print("Unable to find the path MODELGPATH")
            sys.exit(1)
        gpusetup = os.path.join(
            modelgpath, "platforms", "perlmutter", "setupprlm-gpu.sh")
        prgm = os.path.join(modelgpath, "build-gpu", "src", "SuperPions.exe")
        if os.path.exists(prgm):
            print("Found the executable {}".format(prgm))
        else:
            print("Unable to find the executable {}".format(prgm))
    else:
        prgm = find_program()

    # Create a run directory "name"  if does not exist, and cd to it
    dstack.pushd(data["outputfiletag"], mkdir=True)

    # If nodeid is True then append a random 8 digit hex number
    # to the tag labelling the run. This is so that independent runs using the
    # same inputfile, with different seeds, can be run in the same directory
    oldtag = data["outputfiletag"]
    if nodeid:
        runid = "ffffffff"
        if not dry_run:
            runid = str(uuid.uuid4())[:8]
        data["outputfiletag"] = data["outputfiletag"] + "_{}".format(runid)

    tag = data["outputfiletag"]

    #
    checkinputs()

    # Set the seed and write the inputfile to tag.json
    if seed is None:
        data["seed"] = random.randint(1, 2000000000)
    else:
        data["seed"] = seed
    datatojson()

    #
    # Prepare the shell script
    #
    filenamesh = tag + ".sh"
    filenamestdout = tag + ".stdout"

    fh = open(filenamesh, "w")

    if gpu:
        # 4 A100s per GPU node, one rank per GPU; 64 cores = 128 hardware
        # threads, and -c counts hardware threads
        tasks = int(nnodes * 4)
        cpuspertask = 32
        account = "m3722_g"
        constraint = "gpu"
    else:
        tasks = int(nnodes * 128)
        cpuspertask = int(2 * 128 / (tasks / nnodes))
        account = "m3722"
        constraint = "cpu"
    print("#!/bin/bash", file=fh)
    print("#SBATCH -A {}".format(account), file=fh)
    print("#SBATCH -C {}".format(constraint), file=fh)
    if debug:
        print("#SBATCH --qos debug", file=fh)
        print("#SBATCH -t 00:30:00", file=fh)
    else:
        print("#SBATCH -q regular", file=fh)
        print("#SBATCH -t {}".format(int(math.ceil(time * 60.0))), file=fh)
    print("#SBATCH -N {}".format(nnodes), file=fh)
    print("#SBATCH --ntasks={}".format(tasks), file=fh)
    print("#SBATCH --cpus-per-task={}".format(cpuspertask), file=fh)
    if gpu:
        print("#SBATCH --gpus-per-node=4", file=fh)

    # Set up the shell environment
    print("", file=fh)
    if gpu:
        print("source {}".format(gpusetup), file=fh)
    print("export HDF5_DISABLE_VERSION_CHECK=2", file=fh)
    print("", file=fh)
    print("#run the application:", file=fh)
    print('date  "+%%x %%T" > %s_time.out' % (data["outputfiletag"]), file=fh)
    # Write the command that actually runds the program
    # On GPUs every rank must see all 4 GPUs (--gpu-bind=none): cray-mpich
    # exchanges GPU buffers with CUDA IPC, which hangs with one GPU per rank.
    # PETSc then gives each rank its own GPU.
    gpuopts = "--gpus-per-node=4 --gpu-bind=none " if gpu else ""
    print(
        "srun -n %d --cpu_bind=cores -c %d %s%s -input %s "
        % (tasks, cpuspertask, gpuopts, prgm,
           data["outputfiletag"] + ".json"),
        end=" ",
        file=fh,
    )
    if gpu:
        print("-dm_vec_type kokkos", end=" ", file=fh)
    # This additional options are  added to the srun command
    for opt in moreopts:
        print(opt, end=" ", file=fh)
    print(file=fh)

    # # Do any post processing of the run
    # programpy = find_program(program_name="x2k.py")
    # print("python {} {}.json".format(programpy,data["outputfiletag"]), file=fh)
    # print(file=fh)

    print('date  "+%%x %%T" >> %s_time.out' % (data["outputfiletag"]), file=fh)
    fh.close()

    # Submit the shell script
    if not dry_run:
        if interactive:
            with open(filenamestdout, "w") as outfile:
                subprocess.run(["sh", filenamesh], stdout=outfile, check=True)
        else:
            subprocess.run(["sbatch", filenamesh])

    # There was a side effect that the outputfiletag got modified
    # This should be undone for transparency
    if nodeid:
        data["outputfiletag"] = oldtag
    # return to the root directory
    dstack.popd()

# runs the actual command current value of data  with mpiexec

########################################################################
# runs the program with current value of data  and mpiexec on local
# mac.
########################################################################


def run(
    program_name="SuperPions.exe",
    moreopts=[],
    dry_run=True,
    time=0,
    seed=None,
    ncpus="2",
    log_view=True,
    mpiexec="mpiexec",
):

    prgm = find_program(program_name)
    tag = data["outputfiletag"]

    # Go to the directory
    dstack.pushd(tag, mkdir=True)

    # set the seed and the inputfile
    if seed is None:
        data["seed"] = random.randint(1, 2000000000)
    else:
        data["seed"] = seed

    datatojson()

    # Execute the program
    opts = [mpiexec, "-n", ncpus, prgm, "-input", tag + ".json"]
    if log_view:
        opts.append("-log_view")
    opts.extend(moreopts)
    print(opts)
    if not dry_run:
        subprocess.run(opts)

    # Go back to the working directory
    dstack.popd()


if __name__ == "__main__":
    print(getdefault_filename("foo"))
