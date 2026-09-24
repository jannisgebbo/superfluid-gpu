#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "ModelA.h"
#include "Profiling.h"
#include "Stepper.h"
#include "gitversion.h"
#include "initialize.h"

// Measurer, where the Petsc are included
#include "measurer.h"
#include "measurer_output.h"

void initialize_event(const int &ievent, ModelA *const model,
                      nlohmann::json &inputs) {
  const auto &ahandler = model->data.ahandler;
  std::string initialization = inputs["initialization"];

  if (initialization == "default") {
    // Do a cold start
    if (ievent == 0) {
      model->initialize();
    }
  } else if (initialization == "loadfile") {
    // Load a standardized initial condition from an HDF5 snapshot. The
    // "loadfile" input is the filename stem; the file itself is
    // <loadfile>_save.h5 (same convention as ModelA::write()).
    std::string fname = inputs["loadfile"].get<std::string>();
    model->read(fname);
  } else if (initialization == "randomspins") {
    model->initialize_random_spins();
    model->initialize_gaussian_charges();
  } else if (initialization == "gaussians") {
    model->initialize(initialize_gaussians, &inputs["gaussians"]);
    model->write(inputs["outputfiletag"].get<std::string>() + "_initial");
  } else if (initialization == "spinwaves") {
    model->initialize(initialize_wave_spins, &inputs["spinwaves"]);
    model->write(inputs["outputfiletag"].get<std::string>() + "_initial");
  } else {
    throw std::runtime_error(
        "Unknown initialization type: " + initialization +
        ". Please use one of the following: default, loadfile, randomspins, "
        "gaussians, spinwaves.");
  }
}

void run_event(const int &ievent, ModelA *const model, Stepper *const step,
               nlohmann::json &inputs) {

  const auto &ahandler = model->data.ahandler;
  auto &atime = model->data.atime;

  // Set up logging for PETSc so we can find out how much time
  // each part takes
  int steps = 0;
  PetscLogEvent measurements, stepmonitor, saving;
  PetscLogEventRegister("Measurements", 0, &measurements);
  PetscLogEventRegister("Saving the fields", 0, &saving);
  PetscLogEventRegister("Steps", 0, &stepmonitor);

  // Set filename for the hdf5 ouput file
  std::string filename;
  if (ahandler.eventmode) {
    std::stringstream namestream;
    namestream << ahandler.outputfiletag << "_" << std::setw(4)
               << std::setfill('0') << ahandler.current_event << ".h5";
    filename = namestream.str();
  } else {
    filename = ahandler.outputfiletag + ".h5";
  }
  PetscFileMode file_access = FILE_MODE_WRITE;
  // Open the file and create the measurement object
  Measurer measurer(model);

  // Creat the measurer output object, which will write the measurements
  // to the h5 file.
  int rank = -1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::unique_ptr<measurer_output_fasthdf5> measurer_output;
  if (rank == 0) {
    measurer_output = std::make_unique<measurer_output_fasthdf5>(
        &measurer, filename, file_access);
  }

  // With -profile_refcopy, time one VecCopy of the solution per step. It moves
  // the same volume as the owned-block copy in DMGlobalToLocal, as a
  // reference for the GhostExchange time.
  const auto &prof = Profiling::get();
  PetscBool refcopy = PETSC_FALSE;
  PetscOptionsHasName(NULL, NULL, "-profile_refcopy", &refcopy);
  Vec refvec = NULL;
  if (refcopy) {
    VecDuplicate(model->solution, &refvec);
  }

  // Start the loop
  const double tiny = 1.e-10;
  while (atime.t() < atime.tfinal() - tiny) {

    // measure the solution every saveFrequency
    if (steps % ahandler.saveFrequency == 0) {
      PetscLogStagePush(prof.measure_stage);
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure(&model->solution);
      if (rank == 0) {
        measurer_output->save();
      }
      PetscPrintf(PETSC_COMM_WORLD,
                  "Event/Timestep %d/%d: step size = %g, time = %g, final = "
                  "%g\n",
                  ahandler.current_event, steps, (double)atime.dt(),
                  (double)atime.t(), (double)atime.tfinal());
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
      PetscLogStagePop();
    }

    // Write the solution to tape if writeFrequency > 0. This is used for
    // plotting of the solution. It is normally not analyzed, or written.
    if (ahandler.writeFrequency > 0 and steps % ahandler.writeFrequency == 0) {
      PetscLogEventBegin(saving, 0, 0, 0, 0);
      std::ostringstream tString;
      // Set the precision to 2 digits after the decimal point filled with 0
      tString << std::fixed << std::setprecision(2) << "_t_" << atime.t();
      model->write(ahandler.outputfiletag + tString.str());
      PetscLogEventEnd(saving, 0, 0, 0, 0);
    }

    // Do the actual steps
    PetscLogEventBegin(stepmonitor, 0, 0, 0, 0);
    step->step(atime.dt());
    PetscLogEventEnd(stepmonitor, 0, 0, 0, 0);

    if (refcopy) {
      PetscLogEventBegin(prof.ref_copy, 0, 0, 0, 0);
      VecCopy(model->solution, refvec);
      PetscLogEventEnd(prof.ref_copy, 0, 0, 0, 0);
    }

    // Increment the clock
    steps++;
    atime += atime.dt();
  }
  VecDestroy(&refvec);
}

void Run(nlohmann::json &inputs) {
  // Digest the inputs, some fields may be modified on ouptut
  ModelAData inputdata(inputs);

  // allocate the grid and initialize
  ModelA model(inputdata);

  // Construct the stepper
  std::unique_ptr<Stepper> step;
  auto &etype = inputdata.ahandler.evolverType;
  if (etype == "SuperSplitStep") {
    std::string step_sequence =
        inputs["SuperSplitStep"].value<std::string>("step_sequence", "ABBBBBB");
    PetscPrintf(PETSC_COMM_WORLD,
                "Using the SuperSplitStep stepper step_sequence %s \n",
                step_sequence.c_str());
    step = std::make_unique<SuperSplitStep>(model, step_sequence);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unrecognized stepper type %s. Aborting...\n",
                etype.c_str());
    return;
  }

  auto &ahandler = model.data.ahandler;
  auto &atime = model.data.atime;
  int nevents = std::max(ahandler.nevents, 1);
  for (int i = 0; i < nevents; i++) {
    atime.reset();
    initialize_event(i, &model, inputs);
    run_event(i, &model, step.get(), inputs);
    ahandler.current_event++;
  }

  // Destroy everything
  step->finalize();
  model.finalize();
}

int main(int argc, char **argv) {

  // Initialization of PETSc universe
  PetscErrorCode ierr;
  std::string help =
      "Usage: \n\n\t ./SuperPions.exe -input input.json [options]\n\n";
  ierr = PetscInitialize(&argc, &argv, (char *)0, help.c_str());
  if (ierr) {
    return ierr;
  }

  // Open the input file and parse the inputs into the ModelAData
  char filename[PETSC_MAX_PATH_LEN] = "";
  ierr = PetscOptionsGetString(NULL, NULL, "-input", filename, sizeof(filename),
                               NULL);
  nlohmann::json inputs;
  std::ifstream ifs(filename);
  if (ifs) {
    ifs >> inputs;
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unable to open input file %s. Aborting...\n",
                filename);
    return PetscFinalize();
  }
  PetscPrintf(PETSC_COMM_WORLD, "Current version: %s\n", gitversion);

  PetscPrintf(PETSC_COMM_WORLD, "Running SuperPions with input file %s\n",
              filename);

  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank ==0) {
     std::cout << "Input parameters:\n" << inputs.dump(2) << std::endl;
  }

  Run(inputs);

  return PetscFinalize();
}
