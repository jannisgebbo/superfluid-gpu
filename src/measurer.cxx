// First: PETSc requires its Kokkos headers before any other PETSc header
#include "KokkosTypes.h"
#include "measurer.h"
#include "O4AlgebraHelper.h"
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
  if (model->data.ahandler.compute_energy) {
    computeEnergy(fld);
  }

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
  constexpr PetscInt Nphi = ModelAData::Nphi, NA = ModelAData::NA;
  double phi[Nphi], rho[NA + ModelAData::NV];
  double phi2 = 0.;
  for (int l = 0; l < Nphi; l++) {
    phi[l] = u(k, j, i, l);
    acc.v[l] += phi[l];
    phi2 += phi[l] * phi[l];
  }
  for (int l = 0; l < NA; l++) {
    rho[l] = u(k, j, i, ModelAData::OffsetA + l);      // A
    rho[l + NA] = u(k, j, i, ModelAData::OffsetV + l); // V
    acc.v[ModelAData::OffsetA + l] += rho[l];
    acc.v[ModelAData::OffsetV + l] += rho[l + NA];
  }
  acc.v[ModelAData::Ndof] += phi2;

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
        for (int s = 0; s < ModelAData::Nphi; s++) {
          const PetscScalar mid = u(k, j, i, s);
          const PetscScalar dz = u(k + 1, j, i, s) - mid;
          const PetscScalar dy = u(k, j + 1, i, s) - mid;
          const PetscScalar dx = u(k, j, i + 1, s) - mid;
          g += dz * dz + dy * dy + dx * dx;
        }
        h += u(k, j, i, 0) * H0;
        for (int c = ModelAData::OffsetA; c < ModelAData::Ndof; c++) {
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

    // Energy (optional)
    const auto &data = model->data;
    const auto &coeff = data.acoefficients;
    if (data.ahandler.compute_energy) {
      const PetscReal H0 = coeff.H * coeff.sigmabyf(data.atime.t());
      PetscScalar grad2 = 0, nab2 = 0, hEn = 0;
      energy_terms(u, xs, ys, zs, xm, ym, zm, H0, grad2, nab2, hEn);
      // Reduce after releasing the view below
      DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u);
      DMRestoreLocalVector(da, &localU);
      PetscScalar localEnergy = 0.5 / coeff.chi * nab2 + 0.5 * grad2 - hEn;
      MPI_Reduce(&localEnergy, &energy, 1, MPIU_SCALAR, MPI_SUM, 0,
                 PETSC_COMM_WORLD);
    } else {
      DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u);
      DMRestoreLocalVector(da, &localU);
    }

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
    return;
  }
#endif
  computeSliceAverage(solution);
  computeSliceAveragePhase(solution);
}

////////////////////////////////////////////////////////////////////////
// Topological charge density computation
//
// q(x) = 1/(12 pi^2) eps^{abcd} phi_a (dx phi_b)(dy phi_c)(dz phi_d)
//       = 1/(12 pi^2) det[ phi , dx phi , dy phi , dz phi ]
//
// with centered finite differences d_i phi = (phi(x+e_i) - phi(x-e_i))/(2 h_i)

// Determinant of a 4x4 matrix whose rows are (r0, r1, r2, r3).
// This equals the fully-antisymmetric contraction
//   eps^{abcd} r0_a r1_b r2_c r3_d   with eps^{0123} = +1.
#ifdef SFGPU_KOKKOS
O4_FUNCTION
#else
static inline
#endif
double det4x4(const double r0[4], const double r1[4],
              const double r2[4], const double r3[4]) {
  // 3x3 minor of rows (r1, r2, r3) over columns (c0, c1, c2).
  // Written out explicitly (no lambda) so it compiles on CUDA devices.
  const double m123 = r1[1] * (r2[2] * r3[3] - r2[3] * r3[2]) -
                      r1[2] * (r2[1] * r3[3] - r2[3] * r3[1]) +
                      r1[3] * (r2[1] * r3[2] - r2[2] * r3[1]);
  const double m023 = r1[0] * (r2[2] * r3[3] - r2[3] * r3[2]) -
                      r1[2] * (r2[0] * r3[3] - r2[3] * r3[0]) +
                      r1[3] * (r2[0] * r3[2] - r2[2] * r3[0]);
  const double m013 = r1[0] * (r2[1] * r3[3] - r2[3] * r3[1]) -
                      r1[1] * (r2[0] * r3[3] - r2[3] * r3[0]) +
                      r1[3] * (r2[0] * r3[1] - r2[1] * r3[0]);
  const double m012 = r1[0] * (r2[1] * r3[2] - r2[2] * r3[1]) -
                      r1[1] * (r2[0] * r3[2] - r2[2] * r3[0]) +
                      r1[2] * (r2[0] * r3[1] - r2[1] * r3[0]);
  return r0[0] * m123 - r0[1] * m023 + r0[2] * m013 - r0[3] * m012;
}

#ifdef SFGPU_KOKKOS
// GPU kernel: compute q(x) at each owned site and write into a device buffer
// indexed in natural (lexicographic) order: idx = i + N*j + N*N*k.
// Sites not owned by this rank stay zero so MPI_Reduce(SUM) assembles all.
static PetscErrorCode topcharge_kernel_kokkos(
    DM da, Vec localU, PetscInt N, double prefac,
    double inv2hx, double inv2hy, double inv2hz,
    Kokkos::View<double *> &qdev) {
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  ConstNodeView u;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, localU, &u));

  const PetscInt Nlat = N;
  Kokkos::parallel_for(
      "TopChargeDensity",
      SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
        constexpr int Nphi = ModelAData::Nphi;
        double phi[Nphi], Dx[Nphi], Dy[Nphi], Dz[Nphi];
        for (int a = 0; a < Nphi; a++) {
          phi[a] = u(k, j, i, a);
          Dx[a] = (u(k, j, i + 1, a) - u(k, j, i - 1, a)) * inv2hx;
          Dy[a] = (u(k, j + 1, i, a) - u(k, j - 1, i, a)) * inv2hy;
          Dz[a] = (u(k + 1, j, i, a) - u(k - 1, j, i, a)) * inv2hz;
        }
        const double q = prefac * det4x4(phi, Dx, Dy, Dz);
        const size_t idx = static_cast<size_t>(i) +
                           static_cast<size_t>(Nlat) * j +
                           static_cast<size_t>(Nlat) * Nlat * k;
        qdev(idx) = q;
      });
  Kokkos::fence();

  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#ifdef SFGPU_CUFFT
// GPU kernel: bin |qtilde(k)|^2 into spherical shells |k| ~ m with atomic adds.
// A free function because nvcc rejects device lambdas in private members.
static void topcharge_binning_kokkos(int Nlat, size_t nxh, int nbins,
                                     const Kokkos::View<double *> &qt_dev,
                                     const Kokkos::View<double *> &Sk_dev,
                                     const Kokkos::View<double *> &Ns_dev) {
  const int half = Nlat / 2;
  Kokkos::parallel_for(
      "TopchargeBinning",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {0, 0, 0}, {Nlat, Nlat, static_cast<int>(nxh)}),
      KOKKOS_LAMBDA(int kz, int ky, int kx) {
        const int mz = (kz <= half) ? kz : kz - Nlat;
        const int my = (ky <= half) ? ky : ky - Nlat;
        const int mx = kx;
        const double r = Kokkos::sqrt(
            static_cast<double>(mx * mx + my * my + mz * mz));
        int m = static_cast<int>(Kokkos::floor(r + 0.5));
        if (m >= nbins) m = nbins - 1;
        const size_t idx =
            (static_cast<size_t>(kz) * Nlat + static_cast<size_t>(ky)) * nxh +
            static_cast<size_t>(kx);
        Kokkos::atomic_add(&Sk_dev(m), qt_dev(idx));
        Kokkos::atomic_add(&Ns_dev(m), 1.0);
      });
  Kokkos::fence();
}
#endif

void Measurer::computeTopChargeFourier(Vec *solution) {
  DM &da = model->domain;
  Vec localU;
  DMGetLocalVector(da, &localU);
  ghost_exchange(da, *solution, localU);

  const auto &data = model->data;
  const double hx = data.hX();
  const double hy = data.hY();
  const double hz = data.hZ();
  const double prefac = 1.0 / (12.0 * M_PI * M_PI);

  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

  const size_t Ntot = static_cast<size_t>(N) * N * N;
  std::vector<double> qlocal(Ntot, 0.0);

#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(*solution)) {
    // GPU path: compute q(x) on device, copy to host
    Kokkos::View<double *> qdev("topcharge_q", Ntot);
    Kokkos::deep_copy(qdev, 0.0);
    topcharge_kernel_kokkos(da, localU, N, prefac,
                            1.0 / (2.0 * hx), 1.0 / (2.0 * hy),
                            1.0 / (2.0 * hz), qdev);
    auto qhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qdev);
    std::memcpy(qlocal.data(), qhost.data(), Ntot * sizeof(double));
  } else
#endif
  {
    // Host path
    G_node ***fld;
    DMDAVecGetArrayRead(da, localU, &fld);

    for (int k = izs; k < izs + nz; k++) {
      for (int j = iys; j < iys + ny; j++) {
        for (int i = ixs; i < ixs + nx; i++) {
          double phi[ModelAData::Nphi], Dx[ModelAData::Nphi],
              Dy[ModelAData::Nphi], Dz[ModelAData::Nphi];
          for (int a = 0; a < ModelAData::Nphi; a++) {
            phi[a] = fld[k][j][i].f[a];
            Dx[a] = (fld[k][j][i + 1].f[a] - fld[k][j][i - 1].f[a]) / (2.0 * hx);
            Dy[a] = (fld[k][j + 1][i].f[a] - fld[k][j - 1][i].f[a]) / (2.0 * hy);
            Dz[a] = (fld[k + 1][j][i].f[a] - fld[k - 1][j][i].f[a]) / (2.0 * hz);
          }
          const double q = prefac * det4x4(phi, Dx, Dy, Dz);
          const size_t idx = static_cast<size_t>(i) +
                             static_cast<size_t>(N) * j +
                             static_cast<size_t>(N) * N * k;
          qlocal[idx] = q;
        }
      }
    }

    DMDAVecRestoreArrayRead(da, localU, &fld);
  }

  DMRestoreLocalVector(da, &localU);

  // Assemble the full q(x) volume on rank 0.
  int rank = -1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::vector<double> qfull;
  if (rank == 0) {
    qfull.assign(Ntot, 0.0);
  }
  MPI_Reduce(qlocal.data(), rank == 0 ? qfull.data() : nullptr,
             static_cast<int>(Ntot), MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

  if (rank != 0) {
    return;
  }

  // 3D FFT.  The FFTW/cuFFT dimensions are (n0,n1,n2) = (NZ,NY,NX) so that
  // the x direction is the contiguous (last) index, matching the natural
  // ordering idx = i + N*j + N*N*k used to fill qfull above.
  const size_t nxh = static_cast<size_t>(N / 2 + 1);
  std::vector<std::complex<double>> qtilde(static_cast<size_t>(N) * N * nxh);
  fftw3d->execute(qfull, qtilde);

  // Zero mode / DC component.
  topcharge_zero = qtilde[0];

  // Spherical (azimuthal) average of the power spectrum |qtilde(k)|^2.
  // Modes are binned by |k| into shells of width dk = 2*pi/L.
  std::fill(topcharge_Sk.begin(), topcharge_Sk.end(), 0.0);
  std::fill(topcharge_Nshell.begin(), topcharge_Nshell.end(), 0.0);

#ifdef SFGPU_CUFFT
  // GPU path: run the spherical binning on device with atomic adds.
  // The qtilde array is already on the host after the cuFFT execute+copy.
  // We upload it to the device, bin there, and download the small result.
  {
    const int Nlat = N;
    const int nbins = NtopchargeBins;
    const size_t nk = static_cast<size_t>(N) * N * nxh;

    // Upload qtilde as pairs of doubles (re, im packed contiguously)
    Kokkos::View<double *> qt_dev("qtilde_norm2", nk);
    {
      // Pre-compute |qtilde|^2 on host and upload (cheaper than uploading
      // complex + computing norm on device for this one-shot measurement)
      std::vector<double> norm2(nk);
      for (size_t i = 0; i < nk; i++) norm2[i] = std::norm(qtilde[i]);
      auto qt_host = Kokkos::View<double *, Kokkos::HostSpace,
                                  Kokkos::MemoryUnmanaged>(norm2.data(), nk);
      Kokkos::deep_copy(qt_dev, qt_host);
    }

    Kokkos::View<double *> Sk_dev("Sk", nbins);
    Kokkos::View<double *> Ns_dev("Nshell", nbins);
    Kokkos::deep_copy(Sk_dev, 0.0);
    Kokkos::deep_copy(Ns_dev, 0.0);

    topcharge_binning_kokkos(Nlat, nxh, nbins, qt_dev, Sk_dev, Ns_dev);

    // Download small result arrays
    auto Sk_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Sk_dev);
    auto Ns_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ns_dev);
    for (int m = 0; m < nbins; m++) {
      topcharge_Nshell[m] = Ns_host(m);
      topcharge_Sk[m] = (Ns_host(m) > 0.0) ? Sk_host(m) / Ns_host(m) : 0.0;
    }
  }
#else
  // CPU path
  {
    const int half = N / 2;
    for (int kz = 0; kz < N; kz++) {
      const int mz = (kz <= half) ? kz : kz - N;
      for (int ky = 0; ky < N; ky++) {
        const int my = (ky <= half) ? ky : ky - N;
        for (int kx = 0; kx < static_cast<int>(nxh); kx++) {
          const int mx = kx;
          const double r =
              sqrt(static_cast<double>(mx * mx + my * my + mz * mz));
          int m = static_cast<int>(floor(r + 0.5));
          if (m >= NtopchargeBins) {
            m = NtopchargeBins - 1;
          }
          const size_t idx =
              (static_cast<size_t>(kz) * N + static_cast<size_t>(ky)) * nxh +
              static_cast<size_t>(kx);
          topcharge_Sk[m] += std::norm(qtilde[idx]);
          topcharge_Nshell[m] += 1.0;
        }
      }
    }
    for (int m = 0; m < NtopchargeBins; m++) {
      if (topcharge_Nshell[m] > 0.0) {
        topcharge_Sk[m] /= topcharge_Nshell[m];
      }
    }
  }
#endif
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
