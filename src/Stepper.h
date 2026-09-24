#ifndef STEPPER_H
#define STEPPER_H

#include "ModelA.h"

class Stepper {
public:
  virtual bool step(const double &dt) = 0;
  virtual void finalize() = 0;
  virtual ~Stepper() = default;
};

/////////////////////////////////////////////////////////////////////////
// Ideal Position Verlet Stepper. Energy is computed before and after the
// step and exposed via getEnergy(), but the step is always accepted.
class IdealPV2 : public Stepper {
public:
  IdealPV2(ModelA &in);
  void finalize() override;
  bool step(const double &dt) override;

  ~IdealPV2() { ; }

  PetscScalar computeEnergy(PetscScalar dt);
  void getEnergy(PetscScalar &oldE, PetscScalar &newE) const {
    oldE = oldEnergy;
    newE = newEnergy;
  }

private:
  ModelA *model;

  DM da;
  const ModelAData &data;

  bool step_no_reject(const double &dt);
  void rotatePhi(G_node ***phinew, double dt);

  PetscScalar oldEnergy;
  PetscScalar newEnergy;

  // Minor optimization
  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
};

/////////////////////////////////////////////////////////////////////////
// Explicit second-order Runge-Kutta step for the deterministic diffusion
// equation of Model G.
class ModelGExplicitDiffusionStep : public Stepper {
public:
  ModelGExplicitDiffusionStep(ModelA &in) : model(&in) { ; }
  bool step(const double &dt) override;
  void finalize() override { ; }
  ~ModelGExplicitDiffusionStep() { ; }

private:
  ModelA *model;
};

/////////////////////////////////////////////////////////////////////////
// Solve the deterministic superfluid equations of motion using operator
// splitting. 'A' = ideal Verlet step, 'B' = explicit diffusion step.
class SuperSplitStep : public Stepper {
public:
  SuperSplitStep(ModelA &in, const std::string &step_sequence = "AB")
      : model(&in), pv2(in), diffusion(in), steps(step_sequence),
        A_count(0), B_count(0) {
    for (char s : steps) {
      if (s == 'A') {
        A_count++;
      } else if (s == 'B') {
        B_count++;
      }
    }
  }
  bool step(const double &dt) override {
    PetscLogEvent ideal_log, diffusive_log;
    PetscLogEventRegister("ExplicitDiffusion", 0, &diffusive_log);
    PetscLogEventRegister("IdealStep", 0, &ideal_log);

    bool ok = true;
    for (char s : steps) {
      if (s == 'A') {
        PetscLogEventBegin(ideal_log, 0, 0, 0, 0);
        ok = ok && pv2.step(dt / A_count);
        PetscLogEventEnd(ideal_log, 0, 0, 0, 0);
      } else if (s == 'B') {
        PetscLogEventBegin(diffusive_log, 0, 0, 0, 0);
        ok = ok && diffusion.step(dt / B_count);
        PetscLogEventEnd(diffusive_log, 0, 0, 0, 0);
      }
    }
    return ok;
  }
  void finalize() override {
    pv2.finalize();
    diffusion.finalize();
  }

private:
  ModelA *model;
  IdealPV2 pv2;
  ModelGExplicitDiffusionStep diffusion;
  std::string steps;  // The steps to take, e.g. "ABBB"
  double A_count = 0; // Number of A steps in the string steps
  double B_count = 0; // Number of B steps in the string steps
};

#endif
