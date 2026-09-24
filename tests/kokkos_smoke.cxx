// Toolchain check for the GPU build (SFGPU_KOKKOS). Builds a periodic 3D DMDA
// with the same layout as the model (10 dof per site, star stencil, width 1)
// on Kokkos vectors, fills it on the device with a known function of the
// global site, does a ghost exchange, and checks every neighbor value
// (including the periodic wrap) on the device.
//
//   srun -n 1 --gpus-per-task=1 ./kokkos_smoke.exe -n 64
//   srun -n 4 --gpus-per-task=1 ./kokkos_smoke.exe -n 128 -log_view
//
// Prints PASS/FAIL and the Vec type, so a silent fallback to host vectors
// shows up.
#include <Kokkos_Core.hpp>
#include <petscdmda_kokkos.hpp>

static const PetscInt Ndof = 10;

// Views of a 3D DMDA vector with dof, indexed (k, j, i, c), in device memory
using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using View4D = Kokkos::Experimental::OffsetView<PetscScalar ****,
                                                Kokkos::LayoutRight, MemSpace>;
using ConstView4D =
    Kokkos::Experimental::OffsetView<const PetscScalar ****,
                                     Kokkos::LayoutRight, MemSpace>;

// Known value at global site (i, j, k), component c; exactly representable
KOKKOS_INLINE_FUNCTION PetscScalar value(PetscInt i, PetscInt j, PetscInt k,
                                         PetscInt c) {
  return (PetscScalar)(c + Ndof * (i + 1000 * (j + 1000 * k)));
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  {
    PetscInt N = 64;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));

    DM da;
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,
                           DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                           DMDA_STENCIL_STAR, N, N, N, PETSC_DECIDE,
                           PETSC_DECIDE, PETSC_DECIDE, Ndof, 1, NULL, NULL,
                           NULL, &da));
    PetscCall(DMSetVecType(da, VECKOKKOS));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));

    Vec u, ul;
    PetscCall(DMCreateGlobalVector(da, &u));
    PetscCall(DMCreateLocalVector(da, &ul));
    VecType vtype;
    PetscCall(VecGetType(u, &vtype));

    PetscInt xs, ys, zs, xm, ym, zm;
    PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>,
                                         Kokkos::IndexType<PetscInt>>;
    Policy owned({zs, ys, xs}, {zs + zm, ys + ym, xs + xm});

    // Fill the owned sites on the device
    {
      View4D uv;
      PetscCall(DMDAVecGetKokkosOffsetViewDOFWrite(da, u, &uv));
      Kokkos::parallel_for(
          "fill", owned, KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
            for (PetscInt c = 0; c < Ndof; c++) {
              uv(k, j, i, c) = value(i, j, k, c);
            }
          });
      PetscCall(DMDAVecRestoreKokkosOffsetViewDOFWrite(da, u, &uv));
    }

    // Ghost exchange (on the device for Kokkos vectors)
    PetscCall(DMGlobalToLocalBegin(da, u, INSERT_VALUES, ul));
    PetscCall(DMGlobalToLocalEnd(da, u, INSERT_VALUES, ul));

    // Check the six neighbors of every owned site against the known function
    PetscInt nbad = 0;
    {
      ConstView4D lv;
      PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, ul, &lv));
      Kokkos::parallel_reduce(
          "check", owned,
          KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i, PetscInt & bad) {
            const PetscInt ip = (i + 1) % N, im = (i + N - 1) % N;
            const PetscInt jp = (j + 1) % N, jm = (j + N - 1) % N;
            const PetscInt kp = (k + 1) % N, km = (k + N - 1) % N;
            for (PetscInt c = 0; c < Ndof; c++) {
              bad += lv(k, j, i + 1, c) != value(ip, j, k, c);
              bad += lv(k, j, i - 1, c) != value(im, j, k, c);
              bad += lv(k, j + 1, i, c) != value(i, jp, k, c);
              bad += lv(k, j - 1, i, c) != value(i, jm, k, c);
              bad += lv(k + 1, j, i, c) != value(i, j, kp, c);
              bad += lv(k - 1, j, i, c) != value(i, j, km, c);
            }
          },
          nbad);
      PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, ul, &lv));
    }

    PetscInt nbad_all = 0;
    PetscCallMPI(
        MPI_Allreduce(&nbad, &nbad_all, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "kokkos_smoke: N=%" PetscInt_FMT
                          " ranks=%d vec=%s exec=%s  %s (%" PetscInt_FMT
                          " bad ghost values)\n",
                          N, size, vtype,
                          Kokkos::DefaultExecutionSpace::name(),
                          nbad_all ? "FAIL" : "PASS", nbad_all));

    PetscCall(VecDestroy(&ul));
    PetscCall(VecDestroy(&u));
    PetscCall(DMDestroy(&da));
  }
  PetscCall(PetscFinalize());
  Kokkos::finalize();
  return 0;
}
