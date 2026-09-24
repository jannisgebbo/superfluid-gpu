// First: PETSc requires its Kokkos headers before any other PETSc header
#include "KokkosTypes.h"
#include "measurer.h"
#include "Profiling.h"
#include "nvector.h"
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

////////////////////////////////////////////////////////////////////////

void Measurer::computeSliceAverage(Vec *solution) {
  // Get the local information and store in info

  DM &da = model->domain;
  Vec localU;
  DMGetLocalVector(da, &localU);
  // take the global vector U and distribute to the local vector localU
  ghost_exchange(da, *solution, localU);

  // From the vector define the pointer for the field phi
  G_node ***fld;
  DMDAVecGetArrayRead(da, localU, &fld);

  // Set up the slize averages initialized to zero in c++11
  std::fill(wallX.v.begin(), wallX.v.end(), 0.);
  std::fill(wallY.v.begin(), wallY.v.end(), 0.);
  std::fill(wallZ.v.begin(), wallZ.v.end(), 0.);

  // Local arrays with same characteristic dimensions
  nvector<PetscScalar, 2> wallLocalX(NObs, N);
  nvector<PetscScalar, 2> wallLocalY(NObs, N);
  nvector<PetscScalar, 2> wallLocalZ(NObs, N);

  std::fill(wallLocalX.v.begin(), wallLocalX.v.end(), 0.);
  std::fill(wallLocalY.v.begin(), wallLocalY.v.end(), 0.);
  std::fill(wallLocalZ.v.begin(), wallLocalZ.v.end(), 0.);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
  PetscReal phi2;

  int actualInd = 0;

  // Store the local averages
  for (int k = izs; k < izs + nz; k++) {
    for (int j = iys; j < iys + ny; j++) {
      for (int i = ixs; i < ixs + nx; i++) {
        phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          wallLocalX(l, i) += fld[k][j][i].f[l];
          wallLocalY(l, j) += fld[k][j][i].f[l];
          wallLocalZ(l, k) += fld[k][j][i].f[l];
          phi2 += pow(fld[k][j][i].f[l], 2);
        }
        for (int l = ModelAData::Nphi; l < ModelAData::Nphi + ModelAData::NA;
             l++) {
          actualInd = l - ModelAData::Nphi;
          wallLocalX(l, i) += fld[k][j][i].A[actualInd];
          wallLocalY(l, j) += fld[k][j][i].A[actualInd];
          wallLocalZ(l, k) += fld[k][j][i].A[actualInd];
        }
        for (int l = ModelAData::Nphi + ModelAData::NA;
             l < ModelAData::Nphi + ModelAData::NA + ModelAData::NV; l++) {
          actualInd = l - ModelAData::Nphi - ModelAData::NA;
          wallLocalX(l, i) += fld[k][j][i].V[actualInd];
          wallLocalY(l, j) += fld[k][j][i].V[actualInd];
          wallLocalZ(l, k) += fld[k][j][i].V[actualInd];
        }
        int l = ModelAData::Nphi + ModelAData::NA + ModelAData::NV;
        wallLocalX(l, i) += phi2;
        wallLocalY(l, j) += phi2;
        wallLocalZ(l, k) += phi2;
      }
    }
  }

  for (int l = 0; l < NObs; ++l) {
    for (int i = 0; i < N; ++i) {
      wallLocalX(l, i) /= PetscReal(N * N);
      wallLocalY(l, i) /= PetscReal(N * N);
      wallLocalZ(l, i) /= PetscReal(N * N);
    }
  }

  // The energy needs the ghost cells filled above
  computeEnergy(fld);

  // Retstore the array
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);

  for (int l = 0; l < NObs; l++) {
    MPI_Reduce(&wallLocalX(l, 0), &wallX(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    MPI_Reduce(&wallLocalY(l, 0), &wallY(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    MPI_Reduce(&wallLocalZ(l, 0), &wallZ(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
  }
}

// fld is a ghosted local array. The sum is reduced to rank 0 in energy.
void Measurer::computeEnergy(G_node ***fld) {
  DM &da = model->domain;
  const auto &data = model->data;
  const auto &coeff = data.acoefficients;
  // Same effective field H * sigma/f as IdealPV2::step_no_reject
  const PetscReal H[4] = {coeff.H * coeff.sigmabyf(data.atime.t()), 0., 0.,
                          0.};

  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

  PetscScalar phimid = 0, grad2 = 0, nab2 = 0, hEn = 0;
  for (PetscInt k = izs; k < izs + nz; k++) {
    for (PetscInt j = iys; j < iys + ny; j++) {
      for (PetscInt i = ixs; i < ixs + nx; i++) {
        for (PetscInt s = 0; s < ModelAData::Nphi; ++s) {
          phimid = fld[k][j][i].f[s];

          grad2 += pow(fld[k + 1][j][i].f[s] - phimid, 2);
          grad2 += pow(fld[k][j + 1][i].f[s] - phimid, 2);
          grad2 += pow(fld[k][j][i + 1].f[s] - phimid, 2);

          hEn += phimid * H[s];
        }
        for (PetscInt s = 0; s < ModelAData::NV; s++) {
          nab2 += pow(fld[k][j][i].V[s], 2);
        }
        for (PetscInt s = 0; s < ModelAData::NA; s++) {
          nab2 += pow(fld[k][j][i].A[s], 2);
        }
      }
    }
  }

  PetscScalar localEnergy = 0.5 / coeff.chi * nab2 + 0.5 * grad2 - hEn;
  MPI_Reduce(&localEnergy, &energy, 1, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);
}

// Helper function for slice averages works for double and complex types
template <typename T>
inline std::array<T, 4> contract_rho(const std::array<T, 4> &n,
                                     const std::array<T, 6> &rho) {
  return {-(n[1] * rho[0]) - n[2] * rho[1] - n[3] * rho[2],
          n[0] * rho[0] + n[3] * rho[4] - n[2] * rho[5],
          n[0] * rho[1] - n[3] * rho[3] + n[1] * rho[5],
          n[0] * rho[2] + n[2] * rho[3] - n[1] * rho[4]};
}

// Helper function for slice averages works for double and complex types
template <typename T>
inline std::array<T, 6> dualize_rho(const std::array<T, 6> &rho) {
  return std::array<T, 6>{rho[3], rho[4], rho[5], rho[0], rho[1], rho[2]};
}

void Measurer::computeSliceAveragePhase(Vec *solution) {
  // Get the local information and store in info

  DM &da = model->domain;
  Vec localU;
  DMGetLocalVector(da, &localU);
  // take the global vector U and distribute to the local vector localU
  ghost_exchange(da, *solution, localU);

  // From the vector define the pointer for the field phi
  G_node ***fld;
  DMDAVecGetArrayRead(da, localU, &fld);

  // Set up the slize averages initialized to zero in c++11
  std::fill(wallXPhase.v.begin(), wallXPhase.v.end(), 0.);
  std::fill(wallYPhase.v.begin(), wallYPhase.v.end(), 0.);
  std::fill(wallZPhase.v.begin(), wallZPhase.v.end(), 0.);

  // Local arrays with same dimensions initialized to zero
  nvector<double, 2> wallXPhaseLocal(NObsPhase, N);
  nvector<double, 2> wallYPhaseLocal(NObsPhase, N);
  nvector<double, 2> wallZPhaseLocal(NObsPhase, N);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

  double sigma;
  std::array<double, 4> n{};
  std::array<double, 4> phi{};
  std::array<double, 6> rho{};
  std::array<double, 4> V{};
  std::array<double, 4> A{};
  // Store the local averages
  for (int k = izs; k < izs + nz; k++) {
    for (int j = iys; j < iys + ny; j++) {
      for (int i = ixs; i < ixs + nx; i++) {

        double phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          phi[l] = fld[k][j][i].f[l];
          phi2 += pow(phi[l], 2);
        }
        sigma = sqrt(phi2);

        if (phi2 > 100. * std::numeric_limits<double>::min()) {
          for (int l = 0; l < ModelAData::Nphi; l++) {
            n[l] = phi[l] / sigma;
          }
        } else {
          n = {1, 0, 0, 0};
        }

        for (int l = 0; l < ModelAData::NA; l++) {
          rho[l] = fld[k][j][i].A[l];
          rho[l + ModelAData::NA] = fld[k][j][i].V[l];
        }
        A = contract_rho(n, rho);
        V = contract_rho(n, dualize_rho(rho));

        wallXPhaseLocal(0, i) += sigma;
        wallYPhaseLocal(0, j) += sigma;
        wallZPhaseLocal(0, k) += sigma;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          wallXPhaseLocal(1 + l, i) += n[l];
          wallYPhaseLocal(1 + l, j) += n[l];
          wallZPhaseLocal(1 + l, k) += n[l];

          wallXPhaseLocal(5 + l, i) += A[l];
          wallYPhaseLocal(5 + l, j) += A[l];
          wallZPhaseLocal(5 + l, k) += A[l];

          wallXPhaseLocal(9 + l, i) += V[l];
          wallYPhaseLocal(9 + l, j) += V[l];
          wallZPhaseLocal(9 + l, k) += V[l];
        }
        wallXPhaseLocal(Measurer::NObsPhase - 1, i) += sigma * sigma;
        wallYPhaseLocal(Measurer::NObsPhase - 1, j) += sigma * sigma;
        wallZPhaseLocal(Measurer::NObsPhase - 1, k) += sigma * sigma;
      }
    }
  }
  for (int l = 0; l < NObsPhase; ++l) {
    for (int i = 0; i < N; ++i) {
      wallXPhaseLocal(l, i) /= PetscReal(N * N);
      wallYPhaseLocal(l, i) /= PetscReal(N * N);
      wallZPhaseLocal(l, i) /= PetscReal(N * N);
    }
  }

  // Retstore the array
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);

  // Bring all the data x data into one
  for (int l = 0; l < NObsPhase; l++) {
    MPI_Reduce(&wallXPhaseLocal(l, 0), &wallXPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&wallYPhaseLocal(l, 0), &wallYPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&wallZPhaseLocal(l, 0), &wallZPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
  }
}

#ifdef SFGPU_KOKKOS
// Per-site observables summed into the slice averages: entries 0..NObs-1
// are X = (phi[4], A[3], V[3], phi2) as in computeSliceAverage, and entries
// NObs..NObs+NObsPhase-1 are (sigma, n[4], A'[4], V'[4], sigma^2) as in
// computeSliceAveragePhase.
struct SliceObs {
  static const int NTot = Measurer::NObs + Measurer::NObsPhase;
  double v[NTot];
  KOKKOS_INLINE_FUNCTION SliceObs() {
    for (int o = 0; o < NTot; o++) v[o] = 0.;
  }
  KOKKOS_INLINE_FUNCTION SliceObs &operator+=(const SliceObs &b) {
    for (int o = 0; o < NTot; o++) v[o] += b.v[o];
    return *this;
  }
};

namespace Kokkos {
template <> struct reduction_identity<SliceObs> {
  KOKKOS_FORCEINLINE_FUNCTION static SliceObs sum() { return SliceObs(); }
};
} // namespace Kokkos

// Adds the observables of site (k, j, i) to acc. Same formulas as the host
// loops (contract_rho and dualize_rho written out).
KOKKOS_INLINE_FUNCTION void add_site_obs(const ConstNodeView &u, PetscInt k,
                                         PetscInt j, PetscInt i,
                                         SliceObs &acc) {
  const int NObs = Measurer::NObs;
  double phi[4], rho[6];
  double phi2 = 0.;
  for (int l = 0; l < 4; l++) {
    phi[l] = u(k, j, i, l);
    acc.v[l] += phi[l];
    phi2 += phi[l] * phi[l];
  }
  for (int l = 0; l < 3; l++) {
    rho[l] = u(k, j, i, 4 + l);     // A
    rho[l + 3] = u(k, j, i, 7 + l); // V
    acc.v[4 + l] += rho[l];
    acc.v[7 + l] += rho[l + 3];
  }
  acc.v[10] += phi2;

  // Rotated to the local direction n = phi / sigma
  const double sigma = Kokkos::sqrt(phi2);
  double n[4] = {1., 0., 0., 0.};
  if (phi2 > 100. * Kokkos::Experimental::norm_min_v<double>) {
    for (int l = 0; l < 4; l++) n[l] = phi[l] / sigma;
  }
  const double *r = rho;
  const double rd[6] = {rho[3], rho[4], rho[5], rho[0], rho[1], rho[2]};
  const double A[4] = {-(n[1] * r[0]) - n[2] * r[1] - n[3] * r[2],
                       n[0] * r[0] + n[3] * r[4] - n[2] * r[5],
                       n[0] * r[1] - n[3] * r[3] + n[1] * r[5],
                       n[0] * r[2] + n[2] * r[3] - n[1] * r[4]};
  const double V[4] = {-(n[1] * rd[0]) - n[2] * rd[1] - n[3] * rd[2],
                       n[0] * rd[0] + n[3] * rd[4] - n[2] * rd[5],
                       n[0] * rd[1] - n[3] * rd[3] + n[1] * rd[5],
                       n[0] * rd[2] + n[2] * rd[3] - n[1] * rd[4]};
  acc.v[NObs + 0] += sigma;
  for (int l = 0; l < 4; l++) {
    acc.v[NObs + 1 + l] += n[l];
    acc.v[NObs + 5 + l] += A[l];
    acc.v[NObs + 9 + l] += V[l];
  }
  acc.v[NObs + Measurer::NObsPhase - 1] += sigma * sigma;
}

// Sums the site observables over each plane of one direction (dir = 0, 1, 2
// for x, y, z walls) with one team per plane. Deterministic: no atomics.
// Returns W(p, o) on the host for the local planes p = 0..nplanes-1.
static Kokkos::View<double **>::host_mirror_type
plane_sums(const ConstNodeView &u, int dir, PetscInt xs, PetscInt ys,
           PetscInt zs, PetscInt xm, PetscInt ym, PetscInt zm) {
  const PetscInt nplanes = dir == 0 ? xm : (dir == 1 ? ym : zm);
  // The two other directions, fastest (contiguous in memory) last
  const PetscInt n1 = dir == 2 ? ym : zm;
  const PetscInt n2 = dir == 0 ? ym : xm;
  Kokkos::View<double **> W("plane_sums", nplanes, SliceObs::NTot);

  using Team = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "SlicePlanes", Kokkos::TeamPolicy<>(nplanes, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Team &team) {
        const PetscInt p = team.league_rank();
        SliceObs sum;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n1 * n2),
            [&](PetscInt idx, SliceObs &acc) {
              const PetscInt a = idx / n2, b = idx % n2;
              PetscInt k, j, i;
              if (dir == 0) {
                k = zs + a, j = ys + b, i = xs + p;
              } else if (dir == 1) {
                k = zs + a, j = ys + p, i = xs + b;
              } else {
                k = zs + p, j = ys + a, i = xs + b;
              }
              add_site_obs(u, k, j, i, acc);
            },
            sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          for (int o = 0; o < SliceObs::NTot; o++) W(p, o) = sum.v[o];
        });
      });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W);
}

// The three sums of the energy, same terms as Measurer::computeEnergy
static void energy_terms(const ConstNodeView &u, PetscInt xs, PetscInt ys,
                         PetscInt zs, PetscInt xm, PetscInt ym, PetscInt zm,
                         PetscReal H0, PetscScalar &grad2, PetscScalar &nab2,
                         PetscScalar &hEn) {
  Kokkos::parallel_reduce(
      "Energy", SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i, PetscScalar & g,
                    PetscScalar & q, PetscScalar & h) {
        for (int s = 0; s < 4; s++) {
          const PetscScalar mid = u(k, j, i, s);
          const PetscScalar dz = u(k + 1, j, i, s) - mid;
          const PetscScalar dy = u(k, j + 1, i, s) - mid;
          const PetscScalar dx = u(k, j, i + 1, s) - mid;
          g += dz * dz + dy * dy + dx * dx;
        }
        h += u(k, j, i, 0) * H0;
        for (int c = 4; c < 10; c++) {
          q += u(k, j, i, c) * u(k, j, i, c);
        }
      },
      grad2, nab2, hEn);
}
#endif

// Sums the local walls of all ranks into global on rank 0
static void reduce_walls(nvector<PetscScalar, 2> &local,
                         nvector<PetscScalar, 2> &global, PetscInt nobs,
                         PetscInt N) {
  for (int l = 0; l < nobs; l++) {
    MPI_Reduce(&local(l, 0), &global(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
  }
}

void Measurer::computeSliceAverages(Vec *solution) {
#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(*solution)) {
    DM &da = model->domain;
    Vec localU;
    DMGetLocalVector(da, &localU);
    // Ghost cells are needed only for the gradient in the energy
    ghost_exchange(da, *solution, localU);

    PetscInt xs, ys, zs, xm, ym, zm;
    DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

    ConstNodeView u;
    DMDAVecGetKokkosOffsetViewDOF(da, localU, &u);

    // Slice averages: plane sums on the device, then the same
    // normalization and reduction as the host loops
    nvector<PetscScalar, 2> wallLocal[3], wallPhaseLocal[3];
    const PetscInt start[3] = {xs, ys, zs};
    for (int dir = 0; dir < 3; dir++) {
      wallLocal[dir].resize(NObs, N);
      wallPhaseLocal[dir].resize(NObsPhase, N);
      std::fill(wallLocal[dir].v.begin(), wallLocal[dir].v.end(), 0.);
      std::fill(wallPhaseLocal[dir].v.begin(), wallPhaseLocal[dir].v.end(),
                0.);
      auto W = plane_sums(u, dir, xs, ys, zs, xm, ym, zm);
      for (PetscInt p = 0; p < (PetscInt)W.extent(0); p++) {
        for (int o = 0; o < NObs; o++) {
          wallLocal[dir](o, start[dir] + p) = W(p, o) / PetscReal(N * N);
        }
        for (int o = 0; o < NObsPhase; o++) {
          wallPhaseLocal[dir](o, start[dir] + p) =
              W(p, NObs + o) / PetscReal(N * N);
        }
      }
    }

    // Energy, same terms as computeEnergy
    const auto &data = model->data;
    const auto &coeff = data.acoefficients;
    const PetscReal H0 = coeff.H * coeff.sigmabyf(data.atime.t());
    PetscScalar grad2 = 0, nab2 = 0, hEn = 0;
    energy_terms(u, xs, ys, zs, xm, ym, zm, H0, grad2, nab2, hEn);

    DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u);
    DMRestoreLocalVector(da, &localU);

    std::fill(wallX.v.begin(), wallX.v.end(), 0.);
    std::fill(wallY.v.begin(), wallY.v.end(), 0.);
    std::fill(wallZ.v.begin(), wallZ.v.end(), 0.);
    std::fill(wallXPhase.v.begin(), wallXPhase.v.end(), 0.);
    std::fill(wallYPhase.v.begin(), wallYPhase.v.end(), 0.);
    std::fill(wallZPhase.v.begin(), wallZPhase.v.end(), 0.);
    reduce_walls(wallLocal[0], wallX, NObs, N);
    reduce_walls(wallLocal[1], wallY, NObs, N);
    reduce_walls(wallLocal[2], wallZ, NObs, N);
    reduce_walls(wallPhaseLocal[0], wallXPhase, NObsPhase, N);
    reduce_walls(wallPhaseLocal[1], wallYPhase, NObsPhase, N);
    reduce_walls(wallPhaseLocal[2], wallZPhase, NObsPhase, N);

    PetscScalar localEnergy = 0.5 / coeff.chi * nab2 + 0.5 * grad2 - hEn;
    MPI_Reduce(&localEnergy, &energy, 1, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    return;
  }
#endif
  computeSliceAverage(solution);
  computeSliceAveragePhase(solution);
}

// Given the fourier transform of phi_a(t,k) and other fields
// stored in the wallk strucutre, compute the rotated fields
// The zero mode has phi(t,0) defines a unit four vector n_a
//
// wallk_rotated contains  sigma = n_a phi_a and pi_b = rho_{ab} n_b
// as well as the axial vector density  A_a = rho_{ab} n_b and
// corresponding vector density V_a. Finally phi2 is also
// stored in this structure
void rotate_to_vev(nvector<std::complex<double>, 2> &wallk,
                   nvector<std::complex<double>, 2> &wallk_rotated) {
  std::array<std::complex<double>, 4> n{};
  std::array<std::complex<double>, 4> phi{};
  std::array<std::complex<double>, 4> phir{};
  std::array<std::complex<double>, 6> rho{};
  std::array<std::complex<double>, 4> V{};
  std::array<std::complex<double>, 4> A{};

  // Extract the vev direction
  double norm = 0.;
  for (int a = 0; a < 4; a++) {
    n[a] = wallk(a, 0); // Extract the zero mode  of phi_a
    norm += std::norm(n[a]);
  }

  norm = sqrt(norm);
  if (norm < std::numeric_limits<double>::min()) {
    n = std::array<std::complex<double>, 4>{1, 0, 0, 0};
  } else {
    for (size_t a = 0; a < 4; a++) {
      n[a] /= norm;
    }
  }

  // Loop over momenta and do the rotation to vev
  for (size_t k = 0; k < wallk.N[1]; k++) {
    for (size_t a = 0; a < 4; a++) {
      phi[a] = wallk(a, k);
    }

    for (size_t ab = 0; ab < 6; ab++) {
      rho[ab] = wallk(4 + ab, k);
    }
    std::complex<double> phi2k = wallk(10, k);

    std::complex<double> phis(0.);
    for (size_t a = 0; a < 4; a++) {
      phis += n[a] * phi[a];
    }
    for (size_t a = 0; a < 4; a++) {
      phir[a] = phi[a] - n[a] * phis;
    }

    A = contract_rho(n, rho);
    V = contract_rho(n, dualize_rho(rho));

    wallk_rotated(0, k) = phis;
    for (size_t a = 0; a < 4; a++) {
      wallk_rotated(1 + a, k) = phir[a];
      wallk_rotated(5 + a, k) = A[a];
      wallk_rotated(9 + a, k) = V[a];
    }
    wallk_rotated(13, k) = phi2k;
  }
}

void Measurer::computeDerivedObs() {
  // NB: the intent is that this is to be called only
  // from the rank=0

  // Initializing the average to zero
  OAverage = std::vector<PetscScalar>(NScalars, 0.);

  // Compute the spatial correlation function of wall averages
  for (int l = 0; l < NObs; l++) {
    for (int i = 0; i < N; i++) {
      OAverage[l] += wallX(l, i);
    }
  }
  // Compute <X>
  for (int l = 0; l < NObs; l++) {
    OAverage[l] /= PetscReal(N);
  }

  fftw->execute(wallX, wallX_k);
  fftw->execute(wallY, wallY_k);
  fftw->execute(wallZ, wallZ_k);

  rotate_to_vev(wallX_k, wallX_k_rotated);
  rotate_to_vev(wallY_k, wallY_k_rotated);
  rotate_to_vev(wallZ_k, wallZ_k_rotated);

  fftw->execute(wallXPhase, wallXPhase_k);
  fftw->execute(wallYPhase, wallYPhase_k);
  fftw->execute(wallZPhase, wallZPhase_k);
}
