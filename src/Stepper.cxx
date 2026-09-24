#include "Stepper.h"
#include "ModelA.h"
#include "O4AlgebraHelper.h"
#include <algorithm>
#include <cstdio>

///////////////////////////////////////////////////////////////////////////

IdealPV2::IdealPV2(ModelA &in)
    : model(&in), da(model->domain), data(model->data) {

  // Store for later use
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);
  oldEnergy = 0.;
  newEnergy = 0.;
}

// Position-Verlet update for the ideal (advective) sector. Linearized about
// phi_0 = f this is a wave equation with omega^2 = v^2 khat^2 + m^2, where
// v^2 = f^2 / chi and khat^2 <= 12 / a^2 for the 3D nearest-neighbor
// Laplacian. Verlet is stable for omega_max * dt < 2, i.e. (for small m)
//     v * dt < a / sqrt(3) ~ 0.577 a
// We conventionally take v^2 = f^2 / chi = 1 and a = 1, so dt < 0.577.
bool IdealPV2::step(const double &dt) {
  oldEnergy = computeEnergy(dt);
  bool success = step_no_reject(dt);
  newEnergy = computeEnergy(dt);
  return success;
}

bool IdealPV2::step_no_reject(const double &dt) {

  // drifts the solution by dt / 2.0
  G_node ***phinew;
  PetscCall(DMDAVecGetArray(da, model->solution, &phinew));
  rotatePhi(phinew, dt / 2.0);
  PetscCall(DMDAVecRestoreArray(da, model->solution, &phinew));

  // Get a local vector with ghost cells
  Vec localU;
  PetscCall(DMGetLocalVector(da, &localU));

  // Fill in the ghost celss with mpicalls
  PetscCall(DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localU));

  // Get Access to arrays with drifted solution
  G_node ***phi;
  PetscCall(DMDAVecGetArrayRead(da, localU, &phi));
  PetscCall(DMDAVecGetArray(da, model->solution, &phinew));

  const auto &coeff = data.acoefficients;
  PetscReal H[4] = {coeff.H, 0., 0., 0.};
  if (data.ahandler.superfluidmode) {
    H[0] *= coeff.sigmabyf(data.atime.t());
  }
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);

  PetscScalar advxx, advyy, advzz;
  PetscInt s1, s2, epsilon;

  // Loop over central elements
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        // First evolve the momenta nab

        G_node &centralPhi = phi[k][j][i];
        G_node &phixplus = phi[k][j][i + 1];
        G_node &phixminus = phi[k][j][i - 1];
        G_node &phiyplus = phi[k][j + 1][i];
        G_node &phiyminus = phi[k][j - 1][i];
        G_node &phizplus = phi[k + 1][j][i];
        G_node &phizminus = phi[k - 1][j][i];

        for (PetscInt l = 0; l < ModelAData::NA; l++) {
          advxx = (-phixplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phixplus.f[l + 1] +
                   centralPhi.f[0] * phixminus.f[l + 1] -
                   phixminus.f[0] * centralPhi.f[l + 1]) *
                  axx;

          advyy = (-phiyplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phiyplus.f[l + 1] +
                   centralPhi.f[0] * phiyminus.f[l + 1] -
                   phiyminus.f[0] * centralPhi.f[l + 1]) *
                  ayy;

          advzz = (-phizplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phizplus.f[l + 1] +
                   centralPhi.f[0] * phizminus.f[l + 1] -
                   phizminus.f[0] * centralPhi.f[l + 1]) *
                  azz;

          phinew[k][j][i].A[l] +=
              dt * (advxx + advyy + advzz - H[0] * centralPhi.f[l + 1]);
        }

        for (PetscInt s = 0; s < ModelAData::NV; s++) {

          s1 = (s + 1) % 3;
          s2 = (s + 2) % 3;
          epsilon = ((PetscScalar)(s - s1) * (s1 - s2) * (s2 - s)) / 2.;
          // advection term with epsilon
          advxx = epsilon *
                  (phixplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phixminus.f[1 + s1] -
                   phixplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phixminus.f[1 + s2]) *
                  axx;

          advyy = epsilon *
                  (phiyplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phiyminus.f[1 + s1] -
                   phiyplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phiyminus.f[1 + s2]) *
                  ayy;

          advzz = epsilon *
                  (phizplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phizminus.f[1 + s1] -
                   phizplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phizminus.f[1 + s2]) *
                  azz;

          phinew[k][j][i].V[s] += dt * (advxx + advyy + advzz);
        }
      }
    }
  }

  // drifts dt / 2.0
  rotatePhi(phinew, dt / 2.0);

  PetscCall(DMDAVecRestoreArray(da, model->solution, &phinew));
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &phi));
  PetscCall(DMRestoreLocalVector(da, &localU));

  return true;
}

void IdealPV2::rotatePhi(G_node ***phinew, double dt) {

  const auto &coeff = data.acoefficients;
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {

        // here we have to convert the charge from the chemical potential
        PetscScalar axialmu[ModelAData::NA], vectormu[ModelAData::NV];

        for (PetscInt s = 0; s < ModelAData::NV; s++) {

          vectormu[s] = -phinew[k][j][i].V[s] / coeff.chi * dt;
        }

        for (PetscInt s = 0; s < ModelAData::NA; s++) {

          axialmu[s] = -phinew[k][j][i].A[s] / coeff.chi * dt;
        }

        O4AlgebraHelper::O4Rotation(vectormu, axialmu, phinew[k][j][i].f);
      }
    }
  }
}

PetscScalar IdealPV2::computeEnergy(double dt) {
  DM da = model->domain;
  // Get a local vector with ghost cells
  Vec localUNew;
  DMGetLocalVector(da, &localUNew);

  // Fill in the ghost celss with mpicalls

  DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localUNew);
  DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localUNew);

  const auto &data = model->data;
  const auto &coeff = data.acoefficients;

  G_node ***phiNew;
  DMDAVecGetArrayRead(da, localUNew, &phiNew);

  const PetscReal H[4] = {coeff.H, 0., 0., 0.};

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // Loop over central elements
  PetscScalar phimid = 0, grad2 = 0, nab2 = 0, hEn = 0;
  PetscScalar localEnergy = 0;

  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        for (PetscInt s = 0; s < ModelAData::Nphi; ++s) {

          phimid = phiNew[k][j][i].f[s];

          grad2 += pow(phiNew[k + 1][j][i].f[s] - phimid, 2);
          grad2 += pow(phiNew[k][j + 1][i].f[s] - phimid, 2);
          grad2 += pow(phiNew[k][j][i + 1].f[s] - phimid, 2);

          hEn += phimid * H[s];
        }
        for (PetscInt s = 0; s < ModelAData::NV; s++) {
          nab2 += pow(phiNew[k][j][i].V[s], 2);
        }

        for (PetscInt s = 0; s < ModelAData::NA; s++) {
          nab2 += pow(phiNew[k][j][i].A[s], 2);
        }
      }
    }
  }

  localEnergy += 0.5 / coeff.chi * nab2;
  localEnergy += 0.5 * grad2;
  localEnergy -= hEn;

  PetscScalar energy = 0;
  MPI_Reduce(&localEnergy, &energy, 1, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);

  DMDAVecRestoreArrayRead(da, localUNew, &phiNew);
  DMRestoreLocalVector(da, &localUNew);

  return energy;
}

void IdealPV2::finalize() { ; }

/////////////////////////////////////////////////////////////////////////

// Explicit forward-Euler step for the deterministic diffusion equations of
// Model G (one ghost exchange, one stencil pass). Each lattice mode is
// multiplied by 1 - dt * Gamma * khat^2 with khat^2 <= 12 (a = 1), so
//     dt * Gamma < 1/12   no mode changes sign (non-oscillatory)
//     dt * Gamma < 1/6    stability
// and likewise with D for the charges. We conventionally keep
// D = Gamma / 3, so the phi bound is the binding constraint.
bool ModelGExplicitDiffusionStep::step(const double &dt) {

  auto &data = model->data;

  // Get a local vector with ghost cells
  DM da = model->domain;
  Vec localU;
  PetscCall(DMGetLocalVector(da, &localU));

  // Fill in the ghost celss with mpicalls
  PetscCall(DMGlobalToLocalBegin(da, model->solution, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, model->solution, INSERT_VALUES, localU));

  // Get Access to arrays with drifted solution
  G_node ***phi;
  PetscCall(DMDAVecGetArrayRead(da, localU, &phi));
  G_node ***phinew;
  PetscCall(DMDAVecGetArray(da, model->solution, &phinew));

  const auto &coeff = data.acoefficients;
  PetscReal H[4] = {coeff.H, 0., 0., 0.};
  if (data.ahandler.superfluidmode) {
    H[0] *= coeff.sigmabyf(data.atime.t());
  }
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);

  PetscReal dtG = dt * coeff.Gamma();
  PetscReal dtD = dt * coeff.D();
  PetscReal f = sqrt(data.f2());

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  // Loop over central elements
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        // First evolve the momenta nab

        G_node &centralPhi = phi[k][j][i];
        G_node &phixplus = phi[k][j][i + 1];
        G_node &phixminus = phi[k][j][i - 1];
        G_node &phiyplus = phi[k][j + 1][i];
        G_node &phiyminus = phi[k][j - 1][i];
        G_node &phizplus = phi[k + 1][j][i];
        G_node &phizminus = phi[k - 1][j][i];

        for (PetscInt l = 0; l < ModelAData::Nphi; l++) {

          // Compute the nabla^2 phi
          phinew[k][j][i].f[l] +=
              dtG * (axx * (phixplus.f[l] + phixminus.f[l]) +
                     ayy * (phiyplus.f[l] + phiyminus.f[l]) +
                     azz * (phizplus.f[l] + phizminus.f[l]) -
                     2. * (axx + ayy + azz) * centralPhi.f[l]);

          // Add the mass term (H[l]  is  H * sigma/f)
          phinew[k][j][i].f[l] += dtG * H[l];
        }
        if (data.ahandler.superfluidmode) {
          // In superfluid mode we need to normalize the first four components
          // of the solution to f2
          G_node::normalize_phi(phinew[k][j][i].f, f);
        }

        for (PetscInt l = 0; l < ModelAData::NA; l++) {
          // Compute the nabla^2 A
          phinew[k][j][i].A[l] +=
              dtD * (axx * (phixplus.A[l] + phixminus.A[l]) +
                     ayy * (phiyplus.A[l] + phiyminus.A[l]) +
                     azz * (phizplus.A[l] + phizminus.A[l]) -
                     2. * (axx + ayy + azz) * centralPhi.A[l]);
        }

        for (PetscInt l = 0; l < ModelAData::NV; l++) {
          phinew[k][j][i].V[l] +=
              dtD * (axx * (phixplus.V[l] + phixminus.V[l]) +
                     ayy * (phiyplus.V[l] + phiyminus.V[l]) +
                     azz * (phizplus.V[l] + phizminus.V[l]) -
                     2. * (axx + ayy + azz) * centralPhi.V[l]);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, model->solution, &phinew));
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &phi));
  PetscCall(DMRestoreLocalVector(da, &localU));

  return true;
}
